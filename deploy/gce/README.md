# Retained GCE deployment

This deployment runs one Flower server VM and four Flower client VMs on Google
Compute Engine. It is intentionally a first-pass deployment: Flower uses
`--insecure` (no TLS or SuperNode authentication).

`compose.yaml` in the repository root is the local distributed Docker runtime
and does not depend on these cloud-specific files.

## Validation status

Live GCE validation is unavailable because cloud credits are unavailable.
The Compose files are still validated statically with
`docker compose config`, and both scripts are checked with `bash -n`. Do not
treat the deployment as production-ready or infer live-cloud validation from
those static checks.

## One-command setup

Open [Google Cloud Shell](https://shell.cloud.google.com/), clone this repository,
and run:

```bash
git clone https://github.com/Linus404/federated-machine-learning.git
cd federated-machine-learning

./deploy/gce/gce-bootstrap.sh --project fedml-501219
```

Optional arguments:

```bash
./deploy/gce/gce-bootstrap.sh \
  --project fedml-501219 \
  --zone europe-west3-a
```

The deployment uses exactly four client VMs to match the application's fixed
four-client strategy contract. The script is idempotent. It enables Compute
Engine and IAP, creates or starts the VMs, configures IAP and Flower firewall
rules, installs Docker correctly, copies your checked-out project to every VM,
creates the public artifacts and private shards, copies exactly one shard to each
client, and starts every Compose stack.

> The initial build and IMDB data preparation take several minutes. Do not stop
> the script while it is running.

## Connect and run training

After bootstrap completes, connect to the server:

```bash
gcloud compute ssh --tunnel-through-iap fml-server \
  --project fedml-501219 \
  --zone europe-west3-a
```

Then run exactly one command on the server:

```bash
cd /opt/federated-machine-learning
./deploy/gce/gce-run.sh
```

`gce-run.sh` creates the local Flower connection, verifies that exactly four
SuperNodes are registered online, and then streams the training run.

## Open the dashboard

From **Cloud Shell**, keep this tunnel open:

```bash
gcloud compute ssh --tunnel-through-iap fml-server \
  --project fedml-501219 \
  --zone europe-west3-a \
  --ssh-flag="-N" \
  --ssh-flag="-L 0.0.0.0:8501:127.0.0.1:8501"
```

Then choose **Web preview → Preview on port 8501**. Binding explicitly to the
IPv4 wildcard address avoids Cloud Shell's `bind [::1]:8501` failure. The
dashboard permits the Cloud Shell preview proxy's WebSocket origin, but remains
reachable only through the VM's loopback interface and the authenticated SSH
tunnel.

From a **local terminal**, use a loopback-only tunnel instead:

```bash
gcloud compute ssh --tunnel-through-iap fml-server \
  --project fedml-501219 \
  --zone europe-west3-a \
  --ssh-flag="-N" \
  --ssh-flag="-L 127.0.0.1:8501:127.0.0.1:8501"
```

Then open <http://127.0.0.1:8501>.

## Check status

```bash
gcloud compute ssh --tunnel-through-iap fml-server \
  --project fedml-501219 \
  --zone europe-west3-a \
  --command='cd /opt/federated-machine-learning && docker compose -f deploy/gce/server.compose.yaml ps'
```

## Stop and delete

Stop the VMs when not in use:

```bash
gcloud compute instances stop fml-server fml-client-0 fml-client-1 fml-client-2 fml-client-3 \
  --project fedml-501219 \
  --zone europe-west3-a
```

Delete the disposable deployment and its IAP/Fleet rules when finished:

```bash
gcloud compute instances delete fml-server fml-client-0 fml-client-1 fml-client-2 fml-client-3 \
  --project fedml-501219 \
  --zone europe-west3-a

gcloud compute firewall-rules delete fml-allow-iap-ssh fml-allow-flower-fleet \
  --project fedml-501219
```

## Manual troubleshooting

The bootstrap script verifies Docker on every host. If an SSH connection fails,
check it with:

```bash
gcloud compute ssh fml-server \
  --project fedml-501219 \
  --zone europe-west3-a \
  --troubleshoot \
  --tunnel-through-iap
```

For detailed Google Cloud and Docker requirements, see [IAP TCP forwarding](https://cloud.google.com/iap/docs/using-tcp-forwarding) and [Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/).
