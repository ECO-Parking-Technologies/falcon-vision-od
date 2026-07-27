"""
Cloudflare Access Authentication Module

Handles authentication for sensor API requests through Cloudflare Access.
Service tokens can be found in:
  Cloudflare Dashboard → Zero Trust → Access → Service Auth → Service Tokens
"""

import requests
from typing import Dict, Optional
import json


class CloudflareAccessClient:
    """Client for making authenticated requests through Cloudflare Access"""

    def __init__(self, client_id: str, client_secret: str):
        """
        Initialize Cloudflare Access client.

        Args:
            client_id: CF-Access-Client-Id (format: "xxx.access")
            client_secret: CF-Access-Client-Secret
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = requests.Session()
        self.session.headers.update({
            'CF-Access-Client-Id': client_id,
            'CF-Access-Client-Secret': client_secret,
            'Content-Type': 'application/json'
        })

    def get(self, url: str, params: Optional[Dict] = None) -> Dict:
        """
        Make authenticated GET request.

        Args:
            url: Full URL to request
            params: Optional query parameters

        Returns:
            JSON response as dictionary

        Raises:
            Exception: If request fails or authentication fails
        """
        response = self.session.get(url, params=params)

        # Check if we got redirected to Cloudflare Access login page
        if response.text.startswith('<!DOCTYPE') or response.text.startswith('<html'):
            raise Exception(
                "Authentication failed - got HTML response (Cloudflare Access login page). "
                "Check your CF-Access-Client-Id and CF-Access-Client-Secret."
            )

        response.raise_for_status()

        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {e}\nResponse: {response.text[:500]}")

    def post(self, url: str, data: Optional[Dict] = None, use_form_data: bool = False) -> Dict:
        """
        Make authenticated POST request.

        Args:
            url: Full URL to request
            data: Optional data to send in body
            use_form_data: If True, send as form data (application/x-www-form-urlencoded)
                          If False, send as JSON (default)

        Returns:
            JSON response as dictionary

        Raises:
            Exception: If request fails or authentication fails
        """
        if use_form_data:
            # Send as form data (application/x-www-form-urlencoded)
            response = self.session.post(url, data=data or {})
        else:
            # Send as JSON (default)
            response = self.session.post(url, json=data or {})

        # Check if we got redirected to Cloudflare Access login page
        if response.text.startswith('<!DOCTYPE') or response.text.startswith('<html'):
            raise Exception(
                "Authentication failed - got HTML response (Cloudflare Access login page). "
                "Check your CF-Access-Client-Id and CF-Access-Client-Secret."
            )

        response.raise_for_status()

        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON response: {e}\nResponse: {response.text[:500]}")

    def test_connection(self, base_url: str) -> bool:
        """
        Test authentication by making a simple request.

        Args:
            base_url: Base URL of the gateway (e.g., "https://legacy.fvg-yaamava-north.private.ecofalcondata.com")

        Returns:
            True if authentication successful, False otherwise
        """
        try:
            # Try to enumerate counter topics as a test
            response = self.post(f"{base_url}/plugin/counter-topic/enum")
            return 'response' in response
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("""
Usage: python cloudflare_auth.py <base_url> <client_id> <client_secret>

Example:
  python cloudflare_auth.py \\
    "https://legacy.fvg-yaamava-north.private.ecofalcondata.com" \\
    "abc123.access" \\
    "supersecretkey"

Cloudflare Access Service Token credentials can be found in:
  Cloudflare Dashboard → Zero Trust → Access → Service Auth → Service Tokens
""")
        sys.exit(1)

    base_url = sys.argv[1]
    client_id = sys.argv[2]
    client_secret = sys.argv[3]

    print(f"\n🔍 Testing Cloudflare Access authentication...\n")
    print(f"URL: {base_url}")
    print(f"Client ID: {client_id[:10]}...\n")

    client = CloudflareAccessClient(client_id, client_secret)

    if client.test_connection(base_url):
        print("\n✅ Authentication successful!")
    else:
        print("\n❌ Authentication failed!")
        sys.exit(1)
