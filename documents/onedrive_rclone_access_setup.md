# Remote access to T29TPG.h5 via rclone and Colab

## Context

The Sentinel-2 HDF5 cube (`T29TPG.h5`, approximately 32 GB) is stored on the ULisboa institutional OneDrive (`ulisboa-my.sharepoint.com`). Direct download to a local machine was unreliably slow. Rather than relocate the same single-threaded download to another machine, the fix was to authenticate against the file with `rclone`, which transfers in multiple resumable chunks, and run the transfer inside Google Colab to take advantage of its network path to Microsoft's EU datacentre.

A plain anonymous HTTP request to the share link (tested with `curl`) confirmed this was not a public "anyone with the link" share: the response was a `403` with `X-Forms_Based_Auth_Required`, indicating the link only resolves inside an authenticated browser session tied to the ULisboa tenant (federated SSO, `FedAuth` cookie). This ruled out lazy, unauthenticated HTTP Range reads against the raw file and meant a real OAuth token was required before any remote access, scripted or otherwise.

## Terminal steps (on the local Mac)

Because Colab has no local browser to complete an interactive OAuth login, the token had to be generated on a machine that does have one, the Mac, and then carried over to Colab manually. The command `rclone authorize "onedrive"` was run in a Mac terminal; this opens a browser window, prompts a normal ULisboa SSO login, and on completion prints a JSON token blob (containing `access_token`, `token_type`, `refresh_token`, `expiry`, and `expires_in`) to stdout. The first attempts to manually highlight and copy this blob out of the wrapped terminal output produced a truncated string (most often missing the final closing quote before the brace), which Colab's `rclone config` then rejected with `unexpected end of JSON input`. The reliable fix was to avoid manual selection entirely: pipe the command to a file with `rclone authorize "onedrive" | tee /tmp/token.txt`, then extract exactly the line starting with `{` and copy it byte-for-byte with `grep -o '{.*}' /tmp/token.txt | pbcopy`. This guaranteed the full, unmodified token reached the clipboard for pasting into Colab.

## Configuring the remote in Colab (non-interactive)

Colab's `!`-prefixed shell cells don't reliably support pasting very long strings into an interactive `input()` prompt (the standard `rclone config` wizard hung or truncated when fed the token this way). The fix was to drive `rclone config create` in `--non-interactive` mode from a Python cell, passing the token as a normal Python string (via `subprocess.run`) rather than through any terminal input box. This mode works as a small state machine: each call returns a JSON object with the next required `Option` and a `State` token, which is fed back into the next call via `--continue --state <state> --result <answer>` until configuration is complete. The steps answered in this project were, in order:

1. `rclone config create --non-interactive onedrive onedrive token <token_json>`, the initial call.
2. `config_refresh_token` ("Already have a token, refresh?"), answered `false` since a fresh token had just been supplied.
3. `config_type` ("Type of connection"), answered `onedrive` (OneDrive Personal or Business), since the source path was a personal OneDrive for Business document library (`/personal/hnathanson_office365_ulisboa_pt/...`), not a team SharePoint site.
4. `config_driveid` ("Select drive you want to use"), answered with the single drive rclone found, `Documentos (business)`.

After the final step, `rclone lsd onedrive:` listed the account's top-level folders, confirming the remote was live.

## Downloading the file

The file's path under the drive root (the drive root corresponds to the "Documents" library, so "Documents" itself is not part of the relative path) is `ML/Final_Project/data/hdf5/T29TPG.h5`. The path was confirmed first with `rclone lsf onedrive:ML/Final_Project/data/hdf5/`, then downloaded with:

```
!rclone copy onedrive:ML/Final_Project/data/hdf5/T29TPG.h5 /content/ -v --stats 15s --stats-one-line
```

The `--progress` flag (which redraws a bar in place using terminal control codes) was avoided in favour of `--stats`, since Colab's output panel does not reliably render in-place redraws and showed a blank or oversized box instead of a usable bar. `--stats 15s --stats-one-line` instead prints a fresh, plain text status line every 15 seconds, giving reliable visibility into transferred bytes and speed.

## Persisting the setup across Colab sessions

Colab runtimes are ephemeral. To avoid repeating the entire OAuth process after a runtime reset, the rclone config file (containing the refresh token, valid for an extended period under normal Microsoft Graph token policy) was copied to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
import shutil
shutil.copy('/root/.config/rclone/rclone.conf', '/content/drive/MyDrive/rclone.conf')
```

In any future session, copying this file back to `/root/.config/rclone/rclone.conf` before running rclone restores the authenticated remote without any further login step. This only preserves credentials, not a partially downloaded file: `/content/` is local to the runtime and is wiped on a full disconnect, so an interrupted download still needs to restart from zero unless the destination is pointed at the mounted Drive instead.
