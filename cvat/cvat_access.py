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


def open_client(host, user, password, cf_access=False):
    if not cf_access:
        from cvat_sdk import make_client
        return make_client(host, credentials=(user, password))
    from cvat_sdk.core.client import Client
    cid = Prompt.ask("[cyan]CF-Access-Client-Id[/cyan]", console=console)
    csecret = Prompt.ask("[cyan]CF-Access-Client-Secret[/cyan]",
                         password=True, console=console)
    # version probe in Client() runs before headers are set; through Access it
    # just warns and continues — harmless
    client = Client(url=host)
    client.api_client.set_default_header("CF-Access-Client-Id", cid)
    client.api_client.set_default_header("CF-Access-Client-Secret", csecret)
    client.login((user, password))
    return client
