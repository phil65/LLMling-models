from __future__ import annotations  # noqa: INP001

import time

import requests


resp = requests.post(
    "https://github.com/login/device/code",
    headers={
        "accept": "application/json",
        "editor-version": "Neovim/0.6.1",
        "editor-plugin-version": "copilot.vim/1.16.0",
        "content-type": "application/json",
        "user-agent": "GithubCopilot/1.155.0",
        "accept-encoding": "gzip,deflate,br",
    },
    data='{"client_id":"Iv1.b507a08c87ecfe98","scope":"read:user"}',
)

# Parse the response json, isolating the device_code, user_code, and verification_uri
resp_json = resp.json()
device_code = resp_json.get("device_code")
user_code = resp_json.get("user_code")
verification_uri = resp_json.get("verification_uri")

# Print the user code and verification uri
print(f"Please visit {verification_uri} and enter code {user_code} to authenticate.")

while True:
    time.sleep(5)

    resp = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={
            "accept": "application/json",
            "editor-version": "Neovim/0.6.1",
            "editor-plugin-version": "copilot.vim/1.16.0",
            "content-type": "application/json",
            "user-agent": "GithubCopilot/1.155.0",
            "accept-encoding": "gzip,deflate,br",
        },
        data=f'{{"client_id":"Iv1.b507a08c87ecfe98","device_code":"{device_code}","grant_type":"urn:ietf:params:oauth:grant-type:device_code"}}',
    )

    # Parse the response json, isolating the access_token
    resp_json = resp.json()
    access_token = resp_json.get("access_token")

    if access_token:
        break
print("Authentication success!")
print("Token:")
print(access_token)
