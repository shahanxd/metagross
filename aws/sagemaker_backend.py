"""SageMaker training-job backend for ``aws/ec2_run.py`` (``--backend sagemaker``).

Runs the same job script as the EC2 path (``aws/jobs/<job>.sh``) as one SageMaker training job in an AWS PyTorch
GPU Deep Learning Container (default instance ``ml.g6.24xlarge``: 4x NVIDIA L4, 96 vCPU):

* **code**: the ``git archive HEAD`` tarball goes to S3 and reaches the container as the ``code`` input channel
  (``/opt/ml/input/data/code/metagross.tgz``). The job's ``ContainerEntrypoint`` (:func:`bootstrap_command`) unpacks
  it and runs ``aws/sagemaker_entry.sh``, which runs the job script unchanged;
* **logs**: ``logs/`` is copied to the checkpoint path every two minutes, so ``status`` can show it from S3 while the
  job runs; the pipeline and training logs are also streamed, tagged, to CloudWatch ``/aws/sagemaker/TrainingJobs``;
* **outputs**: the job's ``JOB_OUTPUTS`` globs are copied to ``<checkpoint path>/out/`` (read by ``fetch``) and to
  ``/opt/ml/model`` (``model.tar.gz``, the fallback ``fetch`` uses when ``out/`` is empty);
* **resume**: ``runs/`` lives on the checkpoint path; ``--resume-from <earlier job>`` delivers that job's ``runs/`` as
  the ``resume`` channel, and ``train_seg --resume`` continues from its ``last.pt``.

S3 layout (bucket ``metagross-<account>-<region>``, shared with the EC2 backend)::

    jobs/<job>/sagemaker/<training-job-name>/code/metagross.tgz
    jobs/<job>/sagemaker/<training-job-name>/checkpoints/{runs,logs,out}/...
    jobs/<job>/sagemaker/<training-job-name>/output/model.tar.gz

The pure helpers (request, IAM policies, names, paths, tar extraction) need no boto3; the functions that call AWS
take a ``boto3.session.Session`` and import botocore lazily, so the unit tests run without AWS.
"""

from __future__ import annotations

import datetime as _dt
import fnmatch
import json
import logging
import re
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import IO, Any, Iterable, Optional

LOG = logging.getLogger("ec2_run.sagemaker")

TAG = {"Key": "project", "Value": "metagross"}
ROLE = "metagross-sagemaker-role"
ROLE_POLICY = "metagross-sagemaker-job"
NAME_PREFIX = "metagross"
# Approved quota (2026-09-30 handoff): "ml.g6.24xlarge for training job usage", 1 instance, ap-south-1.
DEFAULT_INSTANCE = "ml.g6.24xlarge"
# AWS PyTorch training DLC, SageMaker flavour. Registry 763104351884 serves ap-south-1 (sagemaker-python-sdk
# image_uri_config/pytorch.json); tag from aws/deep-learning-containers docs/src/data/pytorch-training/
# 2.10-gpu-sagemaker.yml (GA 2026-01-21, patched until 2027-01-21). Override with --image.
DLC_ACCOUNT = "763104351884"
DLC_REPO = "pytorch-training"
DEFAULT_IMAGE_TAG = "2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker"
DEFAULT_VOLUME_GB = 150  # ML storage volume; ml.g6 has local NVMe, whose fixed capacity SageMaker uses instead
MAX_RUNTIME_S = 5 * 24 * 3600  # SageMaker's default MaxRuntimeInSeconds ceiling (5 days)
ML_ROOT = "/opt/ml"
CKPT_LOCAL = f"{ML_ROOT}/checkpoints"  # SageMaker's default checkpoint LocalPath
CODE_CHANNEL = "code"
CODE_TARBALL = "metagross.tgz"
REPO_IN_CONTAINER = f"{ML_ROOT}/code/metagross"
ROLE_RETRIES, ROLE_RETRY_S = 6, 10.0  # a new execution role can take ~10-60 s to become assumable
ENDED = ("Completed", "Failed", "Stopped")  # TrainingJobStatus values after which output/ is final

# API limits from the botocore SageMaker service model (CreateTrainingJob).
JOB_NAME_RE = re.compile(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}")
ENV_KEY_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
ENV_VALUE_MAX = 512
ENV_MAX_ENTRIES = 100
ENTRYPOINT_STR_MAX = 256
# Environment values are visible to anyone who can DescribeTrainingJob: refuse names that look like secrets. Whole
# '_'-separated tokens are matched, so HF_TOKEN / WANDB_API_KEY are refused but TOKENIZERS_PARALLELISM passes.
SECRET_NAME_TOKENS = frozenset({"TOKEN", "SECRET", "PASSWORD", "PASSWD", "PASS", "PAT", "APIKEY", "KEY", "CREDENTIAL",
                                "CREDENTIALS", "PRIVATE", "AUTH"})
SECRET_NAME_PREFIXES = ("AWS_",)
RESERVED_ENV_PREFIX = "MG_"  # set by the launcher for aws/sagemaker_entry.sh

# A job counts as failed (training job status Failed) when none of its required outputs exists.
JOB_REQUIRED = {"seg": "models/lraspp_offroad5_*.onnx"}
# Per-epoch validation mIoU parsed from the tagged training logs aws/sagemaker_entry.sh streams to CloudWatch
# (train_seg logs "epoch <n>: val mIoU <x> ..."); shown as metrics in the SageMaker console.
JOB_METRICS = {
    "seg": [{"Name": f"{aug}:val_miou", "Regex": rf"\[train_lraspp_offroad5_{aug}\] .*epoch \d+: val mIoU ([0-9.]+)"}
            for aug in ("clean", "robust")],
}
ECR_IMAGE_RE = re.compile(r"(\d{12})\.dkr\.ecr\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?/([^:@]+)(?::[^@]+|@sha256:[0-9a-f]+)")


# ------------------------------------------------------------------------------------------------ pure helpers
def image_uri(region: str, tag: str = DEFAULT_IMAGE_TAG) -> str:
    """AWS PyTorch training DLC URI in ``region``."""
    return f"{DLC_ACCOUNT}.dkr.ecr.{region}.amazonaws.com/{DLC_REPO}:{tag}"


def ecr_repository_arn(image: str) -> str:
    """ECR repository ARN of an ECR image URI (the execution role may pull only from it)."""
    m = ECR_IMAGE_RE.fullmatch(image)
    if not m:
        raise ValueError(f"not an ECR image URI with a tag or digest: {image!r}")
    account, region, repo = m.groups()
    partition = "aws-cn" if image.split("/")[0].endswith(".cn") else "aws"
    return f"arn:{partition}:ecr:{region}:{account}:repository/{repo}"


def training_job_name(job: str, now: _dt.datetime) -> str:
    """``metagross-<job>-<UTC yyyymmdd-hhmmss>``; must match SageMaker's TrainingJobName pattern."""
    name = f"{NAME_PREFIX}-{job}-{now.astimezone(_dt.timezone.utc):%Y%m%d-%H%M%S}"
    if not JOB_NAME_RE.fullmatch(name):
        raise ValueError(f"invalid training job name {name!r} (job names need letters, digits and '-')")
    return name


def job_prefix(job: str, name: str) -> str:
    """S3 key prefix (no trailing slash) holding one training job's code, checkpoints and output."""
    return f"jobs/{job}/sagemaker/{name}"


def looks_secret(name: str) -> bool:
    """True when an environment variable name looks like it carries a credential."""
    upper = name.upper()
    return upper.startswith(SECRET_NAME_PREFIXES) or any(tok in SECRET_NAME_TOKENS for tok in upper.split("_"))


def validate_env(pairs: Iterable[str]) -> dict[str, str]:
    """``KEY=VALUE`` strings -> Environment map; raises ValueError on anything SageMaker or this backend refuses."""
    env: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not ENV_KEY_RE.fullmatch(key):
            raise ValueError(f"--env entries must be KEY=VALUE with KEY matching [A-Za-z_][A-Za-z0-9_]*: {pair!r}")
        if key.startswith(RESERVED_ENV_PREFIX):
            raise ValueError(f"{key}: the {RESERVED_ENV_PREFIX}* variables are set by the launcher")
        if looks_secret(key):
            raise ValueError(f"{key}: looks like a secret; training-job environment variables are readable by "
                             "anyone with sagemaker:DescribeTrainingJob, so it is not passed")
        if len(value) > ENV_VALUE_MAX:
            raise ValueError(f"{key}: value longer than {ENV_VALUE_MAX} characters")
        env[key] = value
    return env


def bootstrap_command(repo_dir: str = REPO_IN_CONTAINER) -> str:
    """Shell run by ``bash -c`` as the container entrypoint: unpack the code channel, hand over to the entry script."""
    tarball = f"{ML_ROOT}/input/data/{CODE_CHANNEL}/{CODE_TARBALL}"
    cmd = (f"set -e; mkdir -p {repo_dir}; tar xzf {tarball} -C {repo_dir}; "
           f"exec bash {repo_dir}/aws/sagemaker_entry.sh")
    if len(cmd) > ENTRYPOINT_STR_MAX:
        raise ValueError(f"bootstrap command is {len(cmd)} > {ENTRYPOINT_STR_MAX} characters")
    return cmd


def create_training_job_request(*, name: str, job: str, role_arn: str, bucket: str, image: str, instance_type: str,
                                volume_gb: int, max_hours: float, env: dict[str, str], output_globs: list[str],
                                resume_from: Optional[str] = None) -> dict[str, Any]:
    """Keyword arguments for ``sagemaker.create_training_job`` (no AWS call; validated against the API limits)."""
    max_s = int(round(max_hours * 3600))
    if not 0 < max_s <= MAX_RUNTIME_S:
        raise ValueError(f"--max-hours must be in (0, {MAX_RUNTIME_S // 3600}]")
    if volume_gb < 1:
        raise ValueError("--disk-gb must be >= 1")
    if resume_from and not JOB_NAME_RE.fullmatch(resume_from):
        raise ValueError(f"--resume-from {resume_from!r} is not a training job name")
    globs = " ".join(output_globs)
    launcher_env = {"MG_JOB": job, "MG_OUTPUT_GLOBS": globs, "MG_MAX_RUNTIME_S": str(max_s)}
    if job in JOB_REQUIRED:
        launcher_env["MG_REQUIRED_OUTPUTS"] = JOB_REQUIRED[job]
    environment = {**env, **launcher_env}
    if len(environment) > ENV_MAX_ENTRIES:
        raise ValueError(f"at most {ENV_MAX_ENTRIES} environment variables")
    too_long = [k for k, v in environment.items() if len(v) > ENV_VALUE_MAX]
    if too_long:
        raise ValueError(f"environment values longer than {ENV_VALUE_MAX} characters: {too_long}")
    prefix = f"s3://{bucket}/{job_prefix(job, name)}"
    channels = [_s3_channel(CODE_CHANNEL, f"{prefix}/code/")]
    if resume_from:
        channels.append(_s3_channel("resume", f"s3://{bucket}/{job_prefix(job, resume_from)}/checkpoints/runs/"))
    algo: dict[str, Any] = {"TrainingImage": image, "TrainingInputMode": "File",
                            "ContainerEntrypoint": ["bash", "-c", bootstrap_command()]}
    if JOB_METRICS.get(job):
        algo["MetricDefinitions"] = JOB_METRICS[job]
    return {
        "TrainingJobName": name,
        "RoleArn": role_arn,
        "AlgorithmSpecification": algo,
        "InputDataConfig": channels,
        "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/jobs/{job}/sagemaker/"},  # -> <name>/output/model.tar.gz
        "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": int(volume_gb)},
        "StoppingCondition": {"MaxRuntimeInSeconds": max_s},
        "CheckpointConfig": {"S3Uri": f"{prefix}/checkpoints/", "LocalPath": CKPT_LOCAL},
        "Environment": environment,
        "EnableNetworkIsolation": False,  # the job downloads the datasets (Hugging Face) and pip packages
        "Tags": [dict(TAG), {"Key": "job", "Value": job}],
    }


def _s3_channel(channel: str, s3_uri: str) -> dict[str, Any]:
    return {"ChannelName": channel, "InputMode": "File", "DataSource": {"S3DataSource": {
        "S3DataType": "S3Prefix", "S3Uri": s3_uri, "S3DataDistributionType": "FullyReplicated"}}}


def role_policies(*, bucket: str, region: str, account: str, image: str) -> tuple[dict, dict]:
    """(trust policy, inline policy) of the SageMaker execution role: this bucket's ``jobs/*/sagemaker/`` prefixes
    (only when the bucket is owned by ``account``: bucket ARNs carry no account, so a same-named bucket elsewhere
    must not match), the job's CloudWatch log group and metrics, and pulls from the training image's ECR repository."""
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    partition = "aws-cn" if region.startswith("cn-") else "aws"
    own_bucket = {"StringEquals": {"s3:ResourceAccount": account}}
    policy = {"Version": "2012-10-17", "Statement": [
        {"Sid": "Bucket", "Effect": "Allow", "Action": ["s3:ListBucket", "s3:GetBucketLocation"],
         "Resource": f"arn:{partition}:s3:::{bucket}", "Condition": own_bucket},
        {"Sid": "JobObjects", "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:AbortMultipartUpload"],
         "Resource": f"arn:{partition}:s3:::{bucket}/jobs/*/sagemaker/*", "Condition": own_bucket},
        {"Sid": "Logs", "Effect": "Allow",
         "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
         "Resource": f"arn:{partition}:logs:{region}:{account}:log-group:/aws/sagemaker/TrainingJobs*"},
        {"Sid": "Metrics", "Effect": "Allow", "Action": "cloudwatch:PutMetricData", "Resource": "*",
         "Condition": {"StringLike": {"cloudwatch:namespace": "/aws/sagemaker/*"}}},
        {"Sid": "EcrAuth", "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*"},
        {"Sid": "EcrPull", "Effect": "Allow",
         "Action": ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"],
         "Resource": ecr_repository_arn(image)},
    ]}
    return trust, policy


def split_s3(uri: str) -> tuple[str, str]:
    """``s3://bucket/key/prefix`` -> (bucket, key prefix)."""
    m = re.fullmatch(r"s3://([^/]+)/?(.*)", uri)
    if not m:
        raise ValueError(f"not an s3:// URI: {uri!r}")
    return m.group(1), m.group(2)


def output_dest(repo: Path, rel: str) -> Path:
    """Local destination of a job output given its repo-relative POSIX path. ``logs/`` goes to ``runs/aws/logs/``
    (as on EC2) so remote logs never mix with local ones. Raises ValueError for absolute or escaping paths, including
    a drive (``D:x``) in any component, which Windows would treat as a jump to another drive."""
    if not rel or rel.startswith("/") or "\\" in rel or "\0" in rel:
        raise ValueError(f"unsafe output path {rel!r}")
    parts = PurePosixPath(rel).parts
    if any(p in ("..", ".") or ":" in p for p in parts):
        raise ValueError(f"unsafe output path {rel!r}")
    if parts[0] == "logs":
        parts = ("runs", "aws") + parts
    dest = repo.joinpath(*parts)
    if dest.parts[:len(repo.parts)] != repo.parts:  # lexical, so a symlinked runs/ is still accepted
        raise ValueError(f"unsafe output path {rel!r}")
    return dest


def matches_globs(rel: str, globs: Iterable[str]) -> bool:
    """True when the repo-relative path matches one of the job's output globs component by component (``*`` never
    crosses ``/``), so ``fetch`` writes only declared outputs, never e.g. ``tests/conftest.py`` or ``.git/hooks``."""
    parts = PurePosixPath(rel).parts
    for glob in globs:
        gparts = PurePosixPath(glob).parts
        if len(gparts) == len(parts) and all(fnmatch.fnmatchcase(p, g) for p, g in zip(parts, gparts)):
            return True
    return False


def extract_outputs(fileobj: IO[bytes], repo: Path, output_globs: Iterable[str]) -> list[Path]:
    """Extract the regular files of a ``model.tar.gz`` that match the job's output globs into the repo via
    :func:`output_dest`; links, devices, undeclared and unsafe paths are skipped (never extracted)."""
    output_globs = list(output_globs)
    got: list[Path] = []
    with tarfile.open(fileobj=fileobj, mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            rel = member.name[2:] if member.name.startswith("./") else member.name
            try:
                dest = output_dest(repo, rel)
            except ValueError:
                LOG.warning("skipping unsafe archive member %r", member.name)
                continue
            if not matches_globs(rel, output_globs):
                LOG.warning("skipping undeclared archive member %r", member.name)
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            got.append(dest)
    return got


def describe_denial(exc: Exception) -> str:
    """One-line reason for a ClientError, flagging AWS Organizations SCP denials (only the org admin can lift)."""
    err = getattr(exc, "response", {}).get("Error", {})
    code, msg = err.get("Code", type(exc).__name__), err.get("Message", str(exc))
    if "service control policy" in msg:
        return f"{code}: explicit deny in an AWS Organizations service control policy (the organisation's " \
               f"management account must allow it; IAM changes in this account cannot)"
    return f"{code}: {msg[:200]}"


# ------------------------------------------------------------------------------------------------ AWS calls
def _client_error() -> type:
    from botocore.exceptions import ClientError  # lazy: the pure helpers above must import without boto3
    return ClientError


def quota(s, instance_type: str) -> Any:
    """Applied SageMaker quota "<type> for training job usage" (instances), or 'unknown (<reason>)'."""
    ClientError = _client_error()
    want = f"{instance_type} for training job usage"
    try:
        for page in s.client("service-quotas").get_paginator("list_service_quotas").paginate(ServiceCode="sagemaker"):
            for q in page.get("Quotas", []):
                if q.get("QuotaName") == want:
                    return q["Value"]
    except ClientError as exc:
        return f"unknown ({describe_denial(exc)})"
    return f"unknown (no quota named {want!r})"


def check(s, instance_type: str = DEFAULT_INSTANCE) -> dict[str, Any]:
    """SageMaker API reachability, training quota for ``instance_type`` and the default training image."""
    ClientError = _client_error()
    out: dict[str, Any] = {"sagemaker_instance_type": instance_type, "sagemaker_image": image_uri(s.region_name)}
    try:
        s.client("sagemaker").list_training_jobs(MaxResults=1)
        out["sagemaker_api"] = "ok"
    except ClientError as exc:
        out["sagemaker_api"] = f"denied ({describe_denial(exc)})"
    out["sagemaker_training_quota"] = quota(s, instance_type)
    return out


def _jobs(s, job: str, status: Optional[str] = None) -> list[str]:
    """This project's training jobs for ``job``, newest first."""
    kw: dict[str, Any] = {"NameContains": f"{NAME_PREFIX}-{job}-", "SortBy": "CreationTime",
                          "SortOrder": "Descending", "MaxResults": 20}
    if status:
        kw["StatusEquals"] = status
    r = s.client("sagemaker").list_training_jobs(**kw)
    return [j["TrainingJobName"] for j in r["TrainingJobSummaries"]
            if j["TrainingJobName"].startswith(f"{NAME_PREFIX}-{job}-")]


def preflight(s, job: str) -> None:
    """Exit with a clear message if the SageMaker API is denied or a training job for ``job`` is already running."""
    ClientError = _client_error()
    try:
        running = _jobs(s, job, status="InProgress")
    except ClientError as exc:
        sys.exit(f"SageMaker API not usable: {describe_denial(exc)}")
    if running:
        sys.exit(f"a {job} training job is already running: {running} (status / terminate --backend sagemaker)")


def ensure_role(s, bucket: str, image: str) -> str:
    """Create or update the execution role; returns its ARN."""
    iam = s.client("iam")
    account = s.client("sts").get_caller_identity()["Account"]
    trust, policy = role_policies(bucket=bucket, region=s.region_name, account=account, image=image)
    try:
        iam.create_role(RoleName=ROLE, AssumeRolePolicyDocument=json.dumps(trust), Tags=[dict(TAG)],
                        Description="metagross SageMaker training jobs (aws/ec2_run.py --backend sagemaker)")
        LOG.info("created role %s", ROLE)
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=ROLE, PolicyDocument=json.dumps(trust))
    iam.put_role_policy(RoleName=ROLE, PolicyName=ROLE_POLICY, PolicyDocument=json.dumps(policy))
    return iam.get_role(RoleName=ROLE)["Role"]["Arn"]


def check_resume(s, job: str, bucket: str, resume_from: str) -> None:
    """Exit unless ``resume_from`` is a training job name with a ``last.pt`` under its ``checkpoints/runs/``: an
    empty File-mode channel would only fail after the instance is provisioned and billed."""
    if not JOB_NAME_RE.fullmatch(resume_from):
        sys.exit(f"--resume-from {resume_from!r} is not a training job name")
    prefix = f"{job_prefix(job, resume_from)}/checkpoints/runs/"
    for page in s.client("s3").get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        if any(o["Key"].endswith("/last.pt") for o in page.get("Contents", [])):
            return
    sys.exit(f"--resume-from {resume_from}: no last.pt under s3://{bucket}/{prefix}")


def launch(s, *, job: str, bucket: str, tarball: bytes, output_globs: list[str], instance_type: str, image: str,
           max_hours: float, volume_gb: int, env: dict[str, str], resume_from: Optional[str] = None,
           now: Optional[_dt.datetime] = None, sleep=time.sleep) -> str:
    """Upload the code, ensure the role and create the training job; returns the training job name.
    ``preflight`` and the bucket must already be done by the caller."""
    ClientError = _client_error()
    name = training_job_name(job, now or _dt.datetime.now(_dt.timezone.utc))
    if resume_from:
        check_resume(s, job, bucket, resume_from)
    role_arn = ensure_role(s, bucket, image)
    req = create_training_job_request(name=name, job=job, role_arn=role_arn, bucket=bucket, image=image,
                                      instance_type=instance_type, volume_gb=volume_gb, max_hours=max_hours,
                                      env=env, output_globs=output_globs, resume_from=resume_from)
    s.client("s3").put_object(Bucket=bucket, Key=f"{job_prefix(job, name)}/code/{CODE_TARBALL}", Body=tarball)
    sm = s.client("sagemaker")
    for attempt in range(ROLE_RETRIES):
        try:
            sm.create_training_job(**req)
            break
        except ClientError as exc:
            err = exc.response.get("Error", {})
            role_not_ready = err.get("Code") == "ValidationException" and "role" in err.get("Message", "").lower()
            if role_not_ready and attempt < ROLE_RETRIES - 1:
                LOG.info("execution role not assumable yet (%s); retrying in %.0f s", err.get("Message"), ROLE_RETRY_S)
                sleep(ROLE_RETRY_S)
                continue
            raise
    LOG.info("created training job %s (%s, %s); hard cap %.1f h; checkpoints/logs -> %s", name, instance_type, image,
             max_hours, req["CheckpointConfig"]["S3Uri"])
    return name


def _utc(t: _dt.datetime) -> _dt.datetime:
    """botocore returns JSON-protocol timestamps (SageMaker) in local time; everything printed here is UTC."""
    return t.astimezone(_dt.timezone.utc)


def resolve(s, job: str, name: Optional[str]) -> str:
    """``name`` or the newest training job for ``job``."""
    if name:
        return name
    names = _jobs(s, job)
    if not names:
        sys.exit(f"no SageMaker training job found for {job!r}")
    return names[0]


def status(s, job: str, name: Optional[str] = None, tail: int = 25) -> None:
    name = resolve(s, job, name)
    d = s.client("sagemaker").describe_training_job(TrainingJobName=name)
    print(f"training job {name} {d['TrainingJobStatus']} / {d.get('SecondaryStatus', '?')} "
          f"({d['ResourceConfig']['InstanceType']}) created {_utc(d['CreationTime']):%Y-%m-%d %H:%M}Z "
          f"billable {d.get('BillableTimeInSeconds', 0) / 3600:.2f} h")
    if d.get("FailureReason"):
        print("failure:", d["FailureReason"])
    for t in d.get("SecondaryStatusTransitions", [])[-4:]:
        print(f"  {_utc(t['StartTime']):%H:%M}Z {t['Status']}: {t.get('StatusMessage', '')[:160]}")
    bucket, prefix = split_s3(d["CheckpointConfig"]["S3Uri"])
    s3 = s.client("s3")
    listing = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/logs/").get("Contents", [])
    for obj in sorted(listing, key=lambda o: o["LastModified"])[-6:]:
        body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode(errors="replace").splitlines()
        print(f"--- {obj['Key'].split('/logs/')[-1]} ({_utc(obj['LastModified']):%H:%M}Z, {len(body)} lines)")
        for line in body[-tail:]:
            print("   ", line[:220])
    if not listing:
        print("(no logs on S3 yet; container stdout is in CloudWatch /aws/sagemaker/TrainingJobs)")


def fetch(s, job: str, repo: Path, output_globs: Iterable[str], name: Optional[str] = None) -> list[Path]:
    """Copy a training job's declared outputs (``output_globs``) into the repo: ``<checkpoints>/out/`` plus the live
    ``logs/``; once the job has ended, falls back to ``model.tar.gz`` when ``out/`` is empty."""
    ClientError = _client_error()
    output_globs = list(output_globs)
    name = resolve(s, job, name)
    d = s.client("sagemaker").describe_training_job(TrainingJobName=name)
    bucket, prefix = split_s3(d["CheckpointConfig"]["S3Uri"])
    prefix = prefix.rstrip("/")
    s3 = s.client("s3")
    got: list[Path] = []
    for sub, keep in (("logs/", "logs/"), ("out/", "")):
        base = f"{prefix}/{sub}"
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=base):
            for obj in page.get("Contents", []):
                rel = keep + obj["Key"][len(base):]
                try:
                    dest = output_dest(repo, rel)
                except ValueError:
                    LOG.warning("skipping unsafe key %s", obj["Key"])
                    continue
                if not matches_globs(rel, output_globs + ["logs/*"]):
                    LOG.warning("skipping undeclared key %s", obj["Key"])
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, obj["Key"], str(dest))
                if sub == "out/":
                    got.append(dest)
    model_uri = d.get("ModelArtifacts", {}).get("S3ModelArtifacts")
    if not got and model_uri and d["TrainingJobStatus"] in ENDED:
        m_bucket, m_key = split_s3(model_uri)
        with tempfile.TemporaryFile() as tmp:
            try:
                s3.download_fileobj(m_bucket, m_key, tmp)
            except ClientError as exc:
                LOG.info("no model.tar.gz for %s (%s)", name, describe_denial(exc))
            else:
                tmp.seek(0)
                got = extract_outputs(tmp, repo, output_globs)
    LOG.info("training job %s is %s; fetched %d output files", name, d["TrainingJobStatus"], len(got))
    for p in got:
        print(" ", p.relative_to(repo))
    return got


def stop(s, job: str, name: Optional[str] = None) -> list[str]:
    """Stop ``name`` or every in-progress training job for ``job``."""
    names = [name] if name else _jobs(s, job, status="InProgress")
    sm = s.client("sagemaker")
    for n in names:
        sm.stop_training_job(TrainingJobName=n)
    LOG.info("stopped %s", names or "nothing (no training job in progress)")
    return names


def cleanup(s, jobs: Iterable[str]) -> None:
    """Stop this project's in-progress training jobs and delete the execution role (the bucket is the caller's)."""
    ClientError = _client_error()
    try:
        for job in jobs:
            stop(s, job)
    except ClientError as exc:
        LOG.warning("could not stop SageMaker jobs: %s", describe_denial(exc))
    iam = s.client("iam")
    try:
        iam.delete_role_policy(RoleName=ROLE, PolicyName=ROLE_POLICY)
        iam.delete_role(RoleName=ROLE)
        LOG.info("deleted role %s", ROLE)
    except ClientError:
        pass
