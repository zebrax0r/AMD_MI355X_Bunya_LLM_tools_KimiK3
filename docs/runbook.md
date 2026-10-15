# Operating runbook

[Documentation home](../README.md) | [Configuration](configuration.md) | [Troubleshooting](troubleshooting.md)

This is the recommended operating path for the configuration selected on
13 October 2026. Historical experiments are not alternative installation steps.

## Existing installations

Keep the existing checkout available for rollback. Do not overwrite
`kimik3.env`, remove caches, rebuild the SIF or download weights just to adopt
this cleanup. No running service is changed by a Git checkout; avoid changing
files underneath an active operational session anyway. Review a separate
checkout and switch at a normal maintenance boundary.

The launcher, batch wrapper, benchmark runner and both client setup scripts are
unchanged. Their known limitations are recorded in [maintenance](maintenance.md).

## 1. Prepare the checkout and configuration

**On the Bunya login node, for a new installation only:**

```bash
cd "/scratch/user/$USER"
git clone https://github.com/zebrax0r/AMD_MI355X_Bunya_LLM_tools_KimiK3.git kimik3-bunya
cd kimik3-bunya
cp -n kimik3-env.example kimik3.env
chmod 600 kimik3.env
${EDITOR:-vi} kimik3.env
```

Set `MODEL_CACHE_DIR` to an absolute scratch path with sufficient quota, and
set `HF_TOKEN` or `HF_TOKEN_FILE`. The example deliberately leaves these local
values empty. Keep the baseline engine settings unless evaluating a separate
configuration. `cp -n` is not a configuration migration.

Load the configuration into the shell for the commands below:

```bash
source ./kimik3.env
: "${MODEL_CACHE_DIR:?Set MODEL_CACHE_DIR in kimik3.env}"
rquota
mkdir -p "$MODEL_CACHE_DIR"
df -h "$MODEL_CACHE_DIR"
```

Budget approximately 1.7 TB for the baseline weights/image. Presharding needs
additional capacity and is not part of this path. `df` does not replace quota
checks.

## 2. Allocate a node and enter it

**Request the allocation from the login node:**

```bash
salloc --partition=gpu_rocm --account=a_rcc --qos=sdf \
       --nodes=1 --gres=gpu:mi355x:8 --ntasks-per-node=1 \
       --cpus-per-task=192 --mem=1800G --time=12:00:00
```

These are the repository's existing site-specific values, not permissions
guaranteed to every Bunya user. Confirm with RCC if your allocation is rejected.

**Before running Apptainer, confirm the shell is on the allocated compute node:**

```bash
hostname -s
printf 'Slurm job: %s\n' "${SLURM_JOB_ID:-not set}"
command -v apptainer
```

If the allocation has left your shell on the login node, enter its compute node
with `srun --jobid "$SLURM_JOB_ID" --pty /bin/bash -l`. Change back to the
checkout's absolute scratch path and run `source ./kimik3.env` in that shell.
Do not run the following container commands on the login node.

## 3. Prepare and check before downloading

**On the allocated compute node, new installations only:**

```bash
./serve-kimik3.sh pull
./serve-kimik3.sh check
./serve-kimik3.sh parsers
./serve-kimik3.sh gpucheck
rquota
df -h "$MODEL_CACHE_DIR"
./serve-kimik3.sh download
```

Do not proceed past a failed or inconclusive compatibility check. Existing
installations with the same image and revisions do not need another download.
A changed image/revision requires a separate qualification and a rollback copy.

## 4. Launch

**On the allocated compute node:**

```bash
./serve-kimik3.sh serve --detach
```

The command waits for readiness and returns the shell. The server remains tied
to the allocation. The existing timeout is 14,400 seconds; the allocation must
have sufficient remaining walltime. Inspect the actual log rather than starting
a second server while the first one is loading:

```bash
tail -f "$MODEL_CACHE_DIR/kimik3-server.log"
```

The existing batch entry point is also retained:

```bash
mkdir -p logs
sbatch serve-kimik3.sbatch
```

Submit from the repository root. The wrapper's script-directory handling has
not been changed in this pass; see [the batch-path caveat](troubleshooting.md#batch-script-cannot-find-the-launcher)
if your Slurm installation executes it from a spool directory. Do not submit a
second GPU job merely to validate a documentation change.

## 5. Verify actual replies and tools

**On the serving node:**

```bash
export KIMIK3_API_KEY="$(cat "$MODEL_CACHE_DIR/kimik3-api-key")"
./serve-kimik3.sh status
./serve-kimik3.sh toolcheck
curl -fsS "http://127.0.0.1:${PORT:-30000}/v1/models" \
  -H "Authorization: Bearer $KIMIK3_API_KEY"
```

Read the reply from a simple chat request and test a real client file-read/edit
cycle. HTTP health alone does not establish coherent output or working tools.
Use [client setup](clients.md) and [validation](validation.md) for those checks.
Do not publish credentials or unreviewed logs.

## 6. Connect a client

Use the actual serving hostname from the launch banner, not a hard-coded node
copied from an old experiment. Follow [client setup](clients.md) for local use
or an SSH tunnel.

## 7. Stop the intended job

From any machine with Slurm access, cancel the specific job with
`scancel <jobid>`. Verify the job ID first.

`./serve-kimik3.sh stop` is the existing local cleanup operation: run it only
on the serving node, with the intended configuration, and no other matching
SGLang server of yours on that host. Its broad process-name cleanup is unchanged.
`status` likewise checks local processes and localhost; it is not a cross-node
status command.
