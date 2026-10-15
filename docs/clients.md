# Client setup

[Documentation home](../README.md) | [Runbook](runbook.md)

The client setup scripts and templates are unchanged. First verify the server
with `status` and `toolcheck` on its allocated node. Use its actual model name,
port, context cap and API key.

## On the serving node

From the checkout, load the existing configuration and key:

```bash
source ./kimik3.env
export KIMIK3_API_KEY="$(cat "$MODEL_CACHE_DIR/kimik3-api-key")"
./opencode-setup.sh --host localhost --port "${PORT:-30000}"
```

Restart opencode and select the Bunya Kimi model. The default generated provider
uses `KIMIK3_API_KEY` from the client's environment. The helper may print the
key in its export instruction: do not paste that output into public logs.

For Kimi Code:

```bash
./kimicode-setup.sh --host localhost --port "${PORT:-30000}"
kimi -m "kimik3-bunya/${SERVED_MODEL_NAME:-kimi-k3}"
```

The existing helper targets the Kimi Code configuration at
`~/.kimi-code/config.toml`, not the legacy Python CLI's `~/.kimi/config.toml`.
It normally writes the key into the config. Review `--no-key` help when using
the environment-based route. The two clients' key-handling mechanisms differ.

## From a laptop

Keep an SSH tunnel open in a separate terminal. Replace `GPU_NODE` with the
hostname in the server banner and `BUNYA_USER` with your Bunya username:

```bash
ssh -N -L 30000:GPU_NODE:30000 BUNYA_USER@bunya1.rcc.uq.edu.au
```

This is the existing login-node forwarding topology. It relies on the current
server's off-node binding; do not change it to loopback-only without changing
the tunnel topology too. Adjust both ports if the server uses another port.

In a checkout on the laptop, provide the API key privately to the environment,
then run the relevant helper with `--host localhost --port 30000`. Pass
`--model` and `--context` explicitly when the server differs from the default:
a laptop config cannot discover the remote server's settings from a local file.

Do not copy your cluster HF token to a client machine just to use the API.
Only the serving API key is needed there. No TLS layer or network-access policy
is added by this cleanup; preserve the existing trusted network/tunnel boundary.

## Configuration caveats

Both helpers read `kimik3.env` beside themselves. Unlike the launcher and
benchmark, they do not consult `KIMIK3_ENV`. For an alternate config, source it
explicitly in the calling shell and use the helpers' explicit overrides; a
conflicting local `kimik3.env` can still change those values.

Do not run a setup helper against an important client config without retaining
a backup. Existing backup/merge behaviour is unchanged, not newly validated
against every installed client version.

## Verify the agent loop

Ask the client to read a disposable file and make a small, reversible edit.
Verify the actual file content, not only the model's claim that it edited it.
Then exercise a growing conversation and inspect streaming/tool behaviour.
The corpus soak test is not a substitute for this end-to-end client check.

Earlier client investigations remain in the
[historical walkthrough](history/README-2026-10-13.md#walkthrough).
