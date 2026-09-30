"""Unit tests for the SageMaker backend of aws/ec2_run.py (no AWS calls; fake boto3 session where needed)."""

from __future__ import annotations

import datetime as dt
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ENTRY = REPO / "aws" / "sagemaker_entry.sh"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


smb = _load("sagemaker_backend", REPO / "aws" / "sagemaker_backend.py")

NOW = dt.datetime(2026, 9, 30, 12, 34, 56, tzinfo=dt.timezone.utc)
BUCKET = "metagross-123456789012-ap-south-1"
IMAGE = smb.image_uri("ap-south-1")
SEG_GLOBS = ["models/lraspp_offroad5_*.onnx", "models/lraspp_offroad5_*.json", "logs/*.log"]


def _request(**kw):
    args = dict(name=smb.training_job_name("seg", NOW), job="seg", role_arn="arn:aws:iam::123456789012:role/r",
                bucket=BUCKET, image=IMAGE, instance_type=smb.DEFAULT_INSTANCE, volume_gb=150, max_hours=12.0,
                env={"EPOCHS": "40"}, output_globs=SEG_GLOBS)
    args.update(kw)
    return smb.create_training_job_request(**args)


# ------------------------------------------------------------------------------------------------ pure helpers
def test_training_job_name_is_utc_and_valid():
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    name = smb.training_job_name("seg", NOW.astimezone(ist))
    assert name == "metagross-seg-20260930-123456"
    assert smb.JOB_NAME_RE.fullmatch(name) and len(name) <= 63
    with pytest.raises(ValueError):
        smb.training_job_name("seg_bad", NOW)


def test_image_uri_and_ecr_arn():
    assert IMAGE == "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:" + smb.DEFAULT_IMAGE_TAG
    assert smb.ecr_repository_arn(IMAGE) == "arn:aws:ecr:ap-south-1:763104351884:repository/pytorch-training"
    digest = "123456789012.dkr.ecr.ap-south-1.amazonaws.com/team/img@sha256:" + "a" * 64
    assert smb.ecr_repository_arn(digest).endswith(":repository/team/img")
    for bad in ("pytorch/pytorch:latest", "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training"):
        with pytest.raises(ValueError):
            smb.ecr_repository_arn(bad)


def test_request_shape_and_limits():
    req = _request()
    name = req["TrainingJobName"]
    assert req["ResourceConfig"] == {"InstanceType": smb.DEFAULT_INSTANCE, "InstanceCount": 1, "VolumeSizeInGB": 150}
    assert smb.DEFAULT_INSTANCE == "ml.g6.16xlarge"  # the approved training-job quota (us-east-1)
    assert req["StoppingCondition"] == {"MaxRuntimeInSeconds": 43200}
    algo = req["AlgorithmSpecification"]
    assert algo["TrainingImage"] == IMAGE and algo["TrainingInputMode"] == "File"
    assert algo["ContainerEntrypoint"][:2] == ["bash", "-c"]
    assert all(len(x) <= smb.ENTRYPOINT_STR_MAX for x in algo["ContainerEntrypoint"])
    assert "aws/sagemaker_entry.sh" in algo["ContainerEntrypoint"][2]
    assert [m["Name"] for m in algo["MetricDefinitions"]] == ["clean:val_miou", "robust:val_miou"]
    (code,) = req["InputDataConfig"]
    assert code["ChannelName"] == "code"
    assert code["DataSource"]["S3DataSource"]["S3Uri"] == f"s3://{BUCKET}/jobs/seg/sagemaker/{name}/code/"
    assert req["CheckpointConfig"] == {"S3Uri": f"s3://{BUCKET}/jobs/seg/sagemaker/{name}/checkpoints/",
                                       "LocalPath": "/opt/ml/checkpoints"}
    assert req["OutputDataConfig"]["S3OutputPath"] == f"s3://{BUCKET}/jobs/seg/sagemaker/"
    env = req["Environment"]
    assert env["EPOCHS"] == "40" and env["MG_JOB"] == "seg" and env["MG_MAX_RUNTIME_S"] == "43200"
    assert env["MG_OUTPUT_GLOBS"].split() == SEG_GLOBS
    assert env["MG_REQUIRED_OUTPUTS"] == smb.JOB_REQUIRED["seg"]
    assert all(smb.ENV_KEY_RE.fullmatch(k) and len(v) <= smb.ENV_VALUE_MAX for k, v in env.items())
    assert req["EnableNetworkIsolation"] is False
    assert {"Key": "project", "Value": "metagross"} in req["Tags"]


def test_request_resume_channel_points_at_earlier_runs():
    req = _request(resume_from="metagross-seg-20260929-000000")
    resume = [c for c in req["InputDataConfig"] if c["ChannelName"] == "resume"]
    assert resume and resume[0]["DataSource"]["S3DataSource"]["S3Uri"] == (
        f"s3://{BUCKET}/jobs/seg/sagemaker/metagross-seg-20260929-000000/checkpoints/runs/")


@pytest.mark.parametrize("hours", [0, -1, 121])
def test_request_rejects_bad_runtime(hours):
    with pytest.raises(ValueError):
        _request(max_hours=hours)


def test_seg_output_globs_fit_one_environment_value():
    ec2_run_src = (REPO / "aws" / "ec2_run.py").read_text(encoding="utf-8")
    block = re.search(r'JOB_OUTPUTS = \{\s*"seg": \[(.*?)\],\s*\}', ec2_run_src, re.S)
    globs = re.findall(r'"([^"]+)"', block.group(1))
    assert globs and len(" ".join(globs)) <= smb.ENV_VALUE_MAX
    req = _request(output_globs=globs)
    assert req["Environment"]["MG_OUTPUT_GLOBS"].split() == globs


def test_metric_regex_matches_tagged_train_log_line():
    line = "[train_lraspp_offroad5_robust] 2026-09-30 12:00:00,001 INFO train_seg: epoch 7: val mIoU 0.6123 acc 0.9 false-safe 0.01"
    robust = [m for m in smb.JOB_METRICS["seg"] if m["Name"] == "robust:val_miou"][0]
    clean = [m for m in smb.JOB_METRICS["seg"] if m["Name"] == "clean:val_miou"][0]
    assert re.search(robust["Regex"], line).group(1) == "0.6123"
    assert re.search(clean["Regex"], line) is None


def test_validate_env():
    assert smb.validate_env(["EPOCHS=2", "TRAIN_SUBSET=512", "IMG=320x416", "DEADLINE="]) == {
        "EPOCHS": "2", "TRAIN_SUBSET": "512", "IMG": "320x416", "DEADLINE": ""}
    assert smb.validate_env(["TOKENIZERS_PARALLELISM=false", "HF_HUB_ENABLE_HF_TRANSFER=1", "TORCH_INDEX=x",
                             "STRICT_TESTS=1", "KEYFRAME_STRIDE=2"])["TOKENIZERS_PARALLELISM"] == "false"
    for bad in ("EPOCHS", "1X=2", "MG_JOB=x", "HF_TOKEN=abc", "AWS_SECRET_ACCESS_KEY=x", "WANDB_API_KEY=x",
                "GH_PAT=x", "DB_PASS=x", "APIKEY=x", "HF_AUTH=x", "aws_region=x", "X=" + "a" * 513):
        with pytest.raises(ValueError):
            smb.validate_env([bad])


def test_role_policies_are_scoped():
    trust, policy = smb.role_policies(bucket=BUCKET, region="ap-south-1", account="123456789012", image=IMAGE)
    assert trust["Statement"][0]["Principal"] == {"Service": "sagemaker.amazonaws.com"}
    by_sid = {st["Sid"]: st for st in policy["Statement"]}
    assert by_sid["JobObjects"]["Resource"] == f"arn:aws:s3:::{BUCKET}/jobs/*/sagemaker/*"
    assert by_sid["Bucket"]["Resource"] == f"arn:aws:s3:::{BUCKET}"
    for sid in ("Bucket", "JobObjects"):  # a same-named bucket in another account must not match
        assert by_sid[sid]["Condition"] == {"StringEquals": {"s3:ResourceAccount": "123456789012"}}
    assert by_sid["EcrPull"]["Resource"] == smb.ecr_repository_arn(IMAGE)
    assert by_sid["Logs"]["Resource"].startswith("arn:aws:logs:ap-south-1:123456789012:log-group:/aws/sagemaker/")
    wildcard = [st["Sid"] for st in policy["Statement"] if st["Resource"] == "*"]
    assert sorted(wildcard) == ["EcrAuth", "Metrics"]  # both are actions without resource-level scoping
    assert all("*" not in str(st["Action"]) for st in policy["Statement"])


def test_output_dest_maps_logs_and_rejects_escapes(tmp_path):
    assert smb.output_dest(tmp_path, "models/a.onnx") == tmp_path / "models" / "a.onnx"
    assert smb.output_dest(tmp_path, "logs/pipeline.log") == tmp_path / "runs" / "aws" / "logs" / "pipeline.log"
    assert smb.output_dest(tmp_path, "./models/a.onnx") == tmp_path / "models" / "a.onnx"  # "." parts collapse
    for bad in ("", "/etc/passwd", "../x", "models/../../x", "a\\b", "C:/x", "models/D:evil.onnx", "logs/Z:x.log",
                "models/a\0.onnx"):
        with pytest.raises(ValueError):
            smb.output_dest(tmp_path, bad)


def test_matches_globs_is_per_component():
    globs = ["models/lraspp_offroad5_*.onnx", "logs/*.log", "runs/seg/lraspp_offroad5_*/metrics.json"]
    assert smb.matches_globs("models/lraspp_offroad5_robust.onnx", globs)
    assert smb.matches_globs("runs/seg/lraspp_offroad5_clean/metrics.json", globs)
    for rel in ("tests/conftest.py", ".git/hooks/pre-commit", "aws/ec2_run.py", "logs/sub/x.log",
                "runs/seg/lraspp_offroad5_x/y/metrics.json", "models/lraspp_offroad5_a.onnx.bak"):
        assert not smb.matches_globs(rel, globs), rel


def _tar(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for info, data in members:
            tf.addfile(info, io.BytesIO(data) if data is not None else None)
    buf.seek(0)
    return buf


def _file(name: str, data: bytes) -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def test_extract_outputs_skips_links_and_escapes(tmp_path):
    link = tarfile.TarInfo("models/evil.onnx")
    link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
    buf = _tar([_file("models/a.onnx", b"onnx"), _file("./logs/pipeline.log", b"done"),
                _file("../escape.txt", b"x"), _file("/abs.txt", b"x"), (link, None),
                _file("tests/conftest.py", b"import os")])
    got = smb.extract_outputs(buf, tmp_path / "repo", ["models/*.onnx", "logs/*.log"])
    assert sorted(p.relative_to(tmp_path / "repo").as_posix() for p in got) == [
        "models/a.onnx", "runs/aws/logs/pipeline.log"]
    assert (tmp_path / "repo" / "models" / "a.onnx").read_bytes() == b"onnx"
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path / "repo" / "models" / "evil.onnx").exists()
    assert not (tmp_path / "repo" / "tests").exists()  # undeclared outputs are never written


def test_describe_denial_flags_scp():
    class Exc(Exception):
        response = {"Error": {"Code": "AccessDeniedException", "Message": "User: arn:aws:iam::1:user/u is not "
                              "authorized to perform: sagemaker:ListTrainingJobs with an explicit deny in a service "
                              "control policy: arn:aws:organizations::2:policy/o-x/service_control_policy/p-y"}}
    msg = smb.describe_denial(Exc())
    assert msg.startswith("AccessDeniedException") and "service control policy" in msg and "management account" in msg


def test_request_validates_against_botocore_model():
    botocore = pytest.importorskip("botocore")
    from botocore.session import get_session
    from botocore.validate import ParamValidator

    model = get_session().get_service_model("sagemaker")
    shape = model.operation_model("CreateTrainingJob").input_shape
    req = _request(resume_from="metagross-seg-20260929-000000")
    report = ParamValidator().validate(req, shape)
    assert not report.has_errors(), report.generate_report()
    assert req["ResourceConfig"]["InstanceType"] in model.shape_for("TrainingInstanceType").enum
    pattern = model.shape_for("TrainingJobName").metadata["pattern"]
    assert re.fullmatch(pattern, req["TrainingJobName"])
    for m in req["AlgorithmSpecification"]["MetricDefinitions"]:
        assert len(m["Regex"]) <= model.shape_for("MetricRegex").metadata["max"]
    assert botocore is not None


# ------------------------------------------------------------------------------------------------ fake AWS flow
class _Exceptions:
    class EntityAlreadyExistsException(Exception):
        pass


class FakeClient:
    def __init__(self, name: str, calls: list, behaviour: dict):
        self.name, self.calls, self.behaviour = name, calls, behaviour
        self.exceptions = _Exceptions

    def __getattr__(self, op):
        def call(*args, **kw):
            self.calls.append((self.name, op, kw or args))
            fn = self.behaviour.get((self.name, op))
            return fn(*args, **kw) if fn else {}
        return call


class FakeSession:
    region_name = "ap-south-1"

    def __init__(self, behaviour: dict | None = None):
        self.calls: list = []
        self.behaviour = behaviour or {}

    def client(self, name: str) -> FakeClient:
        return FakeClient(name, self.calls, self.behaviour)


def test_launch_creates_role_uploads_code_and_retries_role_propagation():
    exc_mod = pytest.importorskip("botocore.exceptions")
    attempts = {"n": 0}

    def create_training_job(**kw):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise exc_mod.ClientError({"Error": {"Code": "ValidationException",
                                                 "Message": "Could not assume role arn:aws:iam::1:role/r"}},
                                      "CreateTrainingJob")
        return {"TrainingJobArn": "arn"}

    s = FakeSession({("sts", "get_caller_identity"): lambda: {"Account": "123456789012"},
                     ("iam", "get_role"): lambda **kw: {"Role": {"Arn": "arn:aws:iam::123456789012:role/" + smb.ROLE}},
                     ("sagemaker", "create_training_job"): create_training_job})
    slept = []
    name = smb.launch(s, job="seg", bucket=BUCKET, tarball=b"tgz", output_globs=SEG_GLOBS,
                      instance_type=smb.DEFAULT_INSTANCE, image=IMAGE, max_hours=1.5, volume_gb=150,
                      env={"EPOCHS": "2"}, now=NOW, sleep=slept.append)
    assert name == "metagross-seg-20260930-123456" and attempts["n"] == 2 and slept == [smb.ROLE_RETRY_S]
    ops = [(c, op) for c, op, _ in s.calls]
    assert ("iam", "create_role") in ops and ("iam", "put_role_policy") in ops
    put = [kw for c, op, kw in s.calls if (c, op) == ("s3", "put_object")][0]
    assert put["Key"] == f"jobs/seg/sagemaker/{name}/code/metagross.tgz" and put["Body"] == b"tgz"
    req = [kw for c, op, kw in s.calls if (c, op) == ("sagemaker", "create_training_job")][-1]
    assert req["RoleArn"].endswith(smb.ROLE) and req["StoppingCondition"]["MaxRuntimeInSeconds"] == 5400


def test_launch_does_not_retry_other_errors():
    exc_mod = pytest.importorskip("botocore.exceptions")

    def create_training_job(**kw):
        raise exc_mod.ClientError({"Error": {"Code": "ResourceLimitExceeded", "Message": "quota 0"}},
                                  "CreateTrainingJob")

    s = FakeSession({("sts", "get_caller_identity"): lambda: {"Account": "123456789012"},
                     ("iam", "get_role"): lambda **kw: {"Role": {"Arn": "arn:aws:iam::1:role/r"}},
                     ("sagemaker", "create_training_job"): create_training_job})
    with pytest.raises(exc_mod.ClientError):
        smb.launch(s, job="seg", bucket=BUCKET, tarball=b"", output_globs=SEG_GLOBS,
                   instance_type=smb.DEFAULT_INSTANCE, image=IMAGE, max_hours=1, volume_gb=150, env={},
                   now=NOW, sleep=lambda _: pytest.fail("must not retry"))


def test_preflight_exits_on_scp_denial_and_running_job():
    exc_mod = pytest.importorskip("botocore.exceptions")

    def denied(**kw):
        raise exc_mod.ClientError({"Error": {"Code": "AccessDeniedException",
                                             "Message": "explicit deny in a service control policy"}},
                                  "ListTrainingJobs")

    with pytest.raises(SystemExit, match="service control policy"):
        smb.preflight(FakeSession({("sagemaker", "list_training_jobs"): denied}), "seg")
    running = {"TrainingJobSummaries": [{"TrainingJobName": "metagross-seg-20260930-000000"},
                                        {"TrainingJobName": "other-metagross-seg-x"}]}
    with pytest.raises(SystemExit, match="already running"):
        smb.preflight(FakeSession({("sagemaker", "list_training_jobs"): lambda **kw: running}), "seg")
    smb.preflight(FakeSession({("sagemaker", "list_training_jobs"): lambda **kw: {"TrainingJobSummaries": []}}), "seg")


class _Paginator:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects

    def paginate(self, Bucket, Prefix):
        yield {"Contents": [{"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)]}


def test_fetch_downloads_out_and_logs(tmp_path):
    name = "metagross-seg-20260930-123456"
    ck = f"jobs/seg/sagemaker/{name}/checkpoints"
    objects = {f"{ck}/out/models/lraspp_offroad5_robust.onnx": b"onnx", f"{ck}/out/logs/pipeline.log": b"done",
               f"{ck}/logs/train_x.log": b"epoch", f"{ck}/runs/seg/x/last.pt": b"ckpt",
               f"{ck}/out/../../evil": b"x", f"{ck}/out/tests/conftest.py": b"import os"}

    def download_file(bucket, key, dest):
        Path(dest).write_bytes(objects[key])

    s = FakeSession({
        ("sagemaker", "describe_training_job"): lambda **kw: {
            "TrainingJobStatus": "Completed", "CheckpointConfig": {"S3Uri": f"s3://{BUCKET}/{ck}/"}},
        ("s3", "get_paginator"): lambda op: _Paginator(objects),
        ("s3", "download_file"): download_file,
    })
    got = smb.fetch(s, "seg", tmp_path, SEG_GLOBS, name=name)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in got) == [
        "models/lraspp_offroad5_robust.onnx", "runs/aws/logs/pipeline.log"]
    assert (tmp_path / "runs" / "aws" / "logs" / "train_x.log").read_bytes() == b"epoch"
    assert not (tmp_path / "runs" / "seg").exists()  # checkpoints are not fetched
    assert not (tmp_path / "tests").exists()  # undeclared keys are skipped


def test_fetch_falls_back_to_model_tarball(tmp_path):
    tarball = _tar([_file("models/lraspp_offroad5_clean.onnx", b"onnx")]).getvalue()

    def download_fileobj(bucket, key, f):
        assert (bucket, key) == (BUCKET, "jobs/seg/sagemaker/n/output/model.tar.gz")
        f.write(tarball)

    s = _fallback_session("Completed", download_fileobj)
    got = smb.fetch(s, "seg", tmp_path, SEG_GLOBS, name="n")
    assert [p.relative_to(tmp_path).as_posix() for p in got] == ["models/lraspp_offroad5_clean.onnx"]


def _fallback_session(status: str, download_fileobj) -> "FakeSession":
    return FakeSession({
        ("sagemaker", "describe_training_job"): lambda **kw: {
            "TrainingJobStatus": status, "CheckpointConfig": {"S3Uri": f"s3://{BUCKET}/jobs/seg/sagemaker/n/checkpoints/"},
            "ModelArtifacts": {"S3ModelArtifacts": f"s3://{BUCKET}/jobs/seg/sagemaker/n/output/model.tar.gz"}},
        ("s3", "get_paginator"): lambda op: _Paginator({}),
        ("s3", "download_fileobj"): download_fileobj,
    })


def test_fetch_fallback_only_for_ended_jobs_and_tolerates_missing_tarball(tmp_path):
    exc_mod = pytest.importorskip("botocore.exceptions")
    s = _fallback_session("InProgress", lambda *a: pytest.fail("no fallback while the job runs"))
    assert smb.fetch(s, "seg", tmp_path, SEG_GLOBS, name="n") == []

    def missing(bucket, key, f):
        raise exc_mod.ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")

    assert smb.fetch(_fallback_session("Failed", missing), "seg", tmp_path, SEG_GLOBS, name="n") == []


def test_check_resume_requires_a_last_pt(tmp_path):
    ck = "jobs/seg/sagemaker/metagross-seg-20260929-000000/checkpoints/runs"
    ok = FakeSession({("s3", "get_paginator"): lambda op: _Paginator({f"{ck}/seg/lraspp_offroad5_clean/last.pt": b""})})
    smb.check_resume(ok, "seg", BUCKET, "metagross-seg-20260929-000000")
    empty = FakeSession({("s3", "get_paginator"): lambda op: _Paginator({f"{ck}/seg/x/train.log": b""})})
    with pytest.raises(SystemExit, match="no last.pt"):
        smb.check_resume(empty, "seg", BUCKET, "metagross-seg-20260929-000000")
    with pytest.raises(SystemExit, match="not a training job name"):
        smb.check_resume(ok, "seg", BUCKET, "../other")


def test_status_prints_utc(capsys):
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    created = dt.datetime(2026, 9, 30, 17, 30, tzinfo=ist)  # 12:00 UTC, as botocore returns it on an IST laptop
    s = FakeSession({
        ("sagemaker", "describe_training_job"): lambda **kw: {
            "TrainingJobStatus": "InProgress", "SecondaryStatus": "Training", "CreationTime": created,
            "ResourceConfig": {"InstanceType": smb.DEFAULT_INSTANCE}, "BillableTimeInSeconds": 3600,
            "SecondaryStatusTransitions": [{"StartTime": created, "Status": "Starting", "StatusMessage": "x"}],
            "CheckpointConfig": {"S3Uri": f"s3://{BUCKET}/jobs/seg/sagemaker/n/checkpoints/"}},
        ("s3", "list_objects_v2"): lambda **kw: {},
    })
    smb.status(s, "seg", name="n")
    out = capsys.readouterr().out
    assert "created 2026-09-30 12:00Z" in out and "12:00Z Starting" in out


# ------------------------------------------------------------------------------------------------ entry script
needs_bash = pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt", reason="needs bash + GNU date")


def _fake_repo(tmp_path: Path, job_body: str) -> Path:
    repo = tmp_path / "ml" / "code" / "metagross"
    (repo / "aws" / "jobs").mkdir(parents=True)
    shutil.copy(ENTRY, repo / "aws" / "sagemaker_entry.sh")
    (repo / "aws" / "jobs" / "fake.sh").write_text("#!/usr/bin/env bash\ncd \"$(dirname \"$0\")/../..\"\n" + job_body)
    return repo


def _run_entry(tmp_path: Path, repo: Path, **env) -> subprocess.CompletedProcess:
    full = {**os.environ, "MG_ML_ROOT": str(tmp_path / "ml"), "MG_JOB": "fake", "MG_FOLLOW_LOGS": "0",
            "MG_LOG_SYNC_S": "1", "MG_MAX_RUNTIME_S": "7200",
            "MG_OUTPUT_GLOBS": "models/*.onnx logs/*.log runs/seg/*/metrics.json",
            "MG_REQUIRED_OUTPUTS": "models/*.onnx", **env}
    for k in ("DEADLINE", "DATA_DIR", "VENV", "SYSTEM_TORCH"):
        if k not in env:
            full.pop(k, None)
    return subprocess.run(["bash", str(repo / "aws" / "sagemaker_entry.sh")], env=full, capture_output=True,
                          text=True, timeout=60)


@needs_bash
def test_entry_runs_job_restores_resume_and_collects_outputs(tmp_path):
    repo = _fake_repo(tmp_path, """set -e
mkdir -p models logs runs/seg/x
echo "$DEADLINE|$DATA_DIR|$VENV|$SYSTEM_TORCH" > models/env.txt
echo onnx > models/m.onnx
echo "pipeline done" > logs/pipeline.log
echo '{}' > runs/seg/x/metrics.json
test -f runs/seg/x/last.pt
""")
    resume = tmp_path / "ml" / "input" / "data" / "resume" / "seg" / "x"
    resume.mkdir(parents=True)
    (resume / "last.pt").write_text("ckpt")
    r = _run_entry(tmp_path, repo)
    assert r.returncode == 0, r.stdout + r.stderr
    ml = tmp_path / "ml"
    assert (repo / "runs").is_symlink() and (ml / "checkpoints" / "runs" / "seg" / "x" / "last.pt").exists()
    for dest in (ml / "checkpoints" / "out", ml / "model"):
        assert (dest / "models" / "m.onnx").read_text().strip() == "onnx"
        assert (dest / "logs" / "pipeline.log").exists()
        assert (dest / "runs" / "seg" / "x" / "metrics.json").exists()
        assert not (dest / "models" / "env.txt").exists()  # only the declared globs are collected
    assert (ml / "checkpoints" / "logs" / "pipeline.log").exists()
    deadline, data_dir, venv, system_torch = (repo / "models" / "env.txt").read_text().strip().split("|")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d", deadline)
    left_min = (dt.datetime.strptime(deadline, "%Y-%m-%dT%H:%M") - dt.datetime.now()).total_seconds() / 60
    assert 90 - 2 <= left_min <= 105 + 1  # 7200 s cap minus a 900 s margin, minute resolution
    assert data_dir == str(ml / "input" / "data" / "scratch" / "data") and venv.endswith("scratch/venv-aws")
    assert system_torch == "1"
    assert not (ml / "output" / "failure").exists()


@needs_bash
def test_entry_propagates_failure_and_writes_reason(tmp_path):
    repo = _fake_repo(tmp_path, "echo 'setup: torch cannot see the GPU'; echo 'boom in training' > logs/pipeline.log; "
                                "exit 2\n")
    r = _run_entry(tmp_path, repo, DEADLINE="2030-01-01T00:00")
    assert r.returncode == 2 and "setup: torch cannot see the GPU" in r.stdout
    failure = (tmp_path / "ml" / "output" / "failure").read_text()
    assert "rc=2" in failure and "torch cannot see the GPU" in failure and "boom in training" in failure
    for d in ("out/logs", "logs"):
        assert (tmp_path / "ml" / "checkpoints" / d / "job.log").exists()
        assert (tmp_path / "ml" / "checkpoints" / d / "pipeline.log").exists()


@needs_bash
def test_entry_fails_when_required_outputs_are_missing(tmp_path):
    repo = _fake_repo(tmp_path, "mkdir -p logs; echo 'pipeline done' > logs/pipeline.log\n")
    r = _run_entry(tmp_path, repo)
    assert r.returncode == 4
    assert "produced none of: models/*.onnx" in (tmp_path / "ml" / "output" / "failure").read_text()


@needs_bash
def test_entry_rejects_unknown_job(tmp_path):
    repo = _fake_repo(tmp_path, "exit 0\n")
    r = _run_entry(tmp_path, repo, MG_JOB="nope")
    assert r.returncode == 3 and "no job script" in r.stdout


# ------------------------------------------------------------------------------------------------ ec2_run.py glue
def _ec2_run():
    pytest.importorskip("boto3")
    return _load("ec2_run", REPO / "aws" / "ec2_run.py")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_code_tarball_is_lf_under_autocrlf_and_refuses_crlf_commits(tmp_path, monkeypatch):
    er = _ec2_run()
    repo = tmp_path / "r"
    (repo / "aws" / "jobs").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "core.autocrlf", "true")  # Git for Windows default: archive would emit CRLF
    (repo / "aws" / "jobs" / "seg.sh").write_bytes(b"#!/usr/bin/env bash\nset -uo pipefail\necho ok\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "lf")
    monkeypatch.setattr(er, "REPO", repo)
    monkeypatch.setattr(er, "GIT_ROOT", repo)
    with tarfile.open(fileobj=io.BytesIO(er.code_tarball()), mode="r:gz") as tf:
        assert b"\r" not in tf.extractfile("aws/jobs/seg.sh").read()
    _git(repo, "config", "core.autocrlf", "false")
    (repo / "aws" / "jobs" / "seg.sh").write_bytes(b"echo crlf\r\n")
    _git(repo, "commit", "-q", "-am", "crlf")
    with pytest.raises(SystemExit, match="CRLF"):
        er.code_tarball()


def test_ensure_bucket_requires_our_account_as_owner():
    er = _ec2_run()
    s = FakeSession({("sts", "get_caller_identity"): lambda: {"Account": "123456789012"}})
    er.ensure_bucket(s, BUCKET)
    head = [kw for c, op, kw in s.calls if (c, op) == ("s3", "head_bucket")]
    assert head == [{"Bucket": BUCKET, "ExpectedBucketOwner": "123456789012"}]


@pytest.mark.parametrize("kw, msg", [({"max_hours": 200.0}, "max-hours"), ({"max_hours": 0.0}, "max-hours"),
                                     ({"resume_from": "../x"}, "not a training job name"),
                                     ({"env": ["HF_TOKEN=x"]}, "secret"), ({"volume_gb": 0}, "disk-gb")])
def test_launch_sagemaker_validates_before_touching_aws(kw, msg):
    er = _ec2_run()

    class NoAws(FakeSession):
        def client(self, name):
            pytest.fail(f"AWS client {name!r} used before validation")

    args = dict(job="seg", itype=smb.DEFAULT_INSTANCE, max_hours=12.0, env=[], volume_gb=150, image=None,
                resume_from=None)
    args.update(kw)
    with pytest.raises(SystemExit, match=msg):
        er.launch_sagemaker(NoAws(), **args)
