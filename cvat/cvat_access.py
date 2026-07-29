"""Shared CVAT client opener with optional Cloudflare Access service token.

Lets the SDK scripts work through a Zero Trust-gated tunnel hostname: create a
service token in Cloudflare (Zero Trust -> Access -> Service Auth), add a
"Service Auth" policy for the CVAT application, then pass --cf-access to the
script and paste the token id/secret when prompted. All credentials are
prompted at runtime and live only in process RAM — never on disk.
"""
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def open_client(host, user, password, cf_access=False, host_header=None):
    """host_header: reach CVAT on the LAN IP while traefik routes by the
    canonical CVAT_HOST name — e.g. --host http://192.168.x.x:8085
    --host-header <tunnel-hostname>. Plain LAN HTTP, no tunnel involved."""
    if not cf_access and not host_header:
        from cvat_sdk import make_client
        return make_client(host, credentials=(user, password))
    from cvat_sdk.core.client import Client
    headers = {}
    if host_header:
        headers["Host"] = host_header
    if cf_access:
        headers["CF-Access-Client-Id"] = Prompt.ask(
            "[cyan]CF-Access-Client-Id[/cyan]", console=console)
        headers["CF-Access-Client-Secret"] = Prompt.ask(
            "[cyan]CF-Access-Client-Secret[/cyan]", password=True,
            console=console)
    # version probe in Client() runs before headers are set; it may warn and
    # continue — harmless
    client = Client(url=host)
    for k, v in headers.items():
        client.api_client.set_default_header(k, v)
    client.login((user, password))
    return client
