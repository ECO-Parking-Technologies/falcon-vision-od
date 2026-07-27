"""Portal API + sensor snapshot classes extracted verbatim from
falcon-vision-ml/inference/dnn/portal_validation_viewer.py (SensorImageFetcher ~L180-420,
PortalAPIClient ~L431-680) for reference while building the unified puller. Not imported.
"""
class SensorImageFetcher:
    """
    Fetch images directly from sensors using Cloudflare Access authentication.

    Uses the sensorSnapshotId from Portal validation data to fetch the EXACT same
    snapshot that was used for the validation, ensuring pixel-perfect image matching.

    ROI modes:
    - 'png': Use pngImage (lossless PNG ROI - exact match to sensor inference)
    - 'jpeg': Use image field (JPEG ROI - lossy compression)
    - 'extract': Get full sensor image and extract ROI ourselves (uses JPEG full image)
    - 'extract_jpeg': Explicitly extract from JPEG full image
    - 'extract_png': Extract from PNG full image (fullPngImage field if available)
    """

    def __init__(self, cf_client_id: str = None, cf_client_secret: str = None,
                 roi_mode: str = 'png'):
        """
        Initialize sensor image fetcher.

        Args:
            cf_client_id: Cloudflare Access Client ID (will prompt if None)
            cf_client_secret: Cloudflare Access Client Secret (will prompt if None)
            roi_mode: How to get ROI - 'png', 'jpeg', 'extract', 'extract_jpeg', or 'extract_png'
        """
        self.cf_client = None
        self.cf_client_id = cf_client_id
        self.cf_client_secret = cf_client_secret
        self.roi_mode = roi_mode
        self._initialized = False
        # In-memory cache: (sensor_url, snapshot_uuid) -> snapshot_data
        self._snapshot_cache = {}
        # Disk cache directory for sensor snapshots
        self._cache_dir = ARTIFACTS_DIR / "validation_image_cache" / "sensor_snapshots"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self, console: Console) -> bool:
        """Initialize Cloudflare client, prompting for credentials if needed."""
        if not CLOUDFLARE_AVAILABLE:
            console.print("[red]Cloudflare auth module not available[/red]")
            return False

        if self._initialized and self.cf_client:
            return True

        # Prompt for credentials if not provided
        if not self.cf_client_id or not self.cf_client_secret:
            console.print()
            console.print(Panel(
                "[bold cyan]Cloudflare Access Authentication[/bold cyan]\n\n"
                "[dim]Required to fetch images directly from sensors.[/dim]\n"
                "[dim]Get credentials from Cloudflare Dashboard → Zero Trust → Access → Service Tokens[/dim]",
                box=box.ROUNDED
            ))
            console.print()

            self.cf_client_id = Prompt.ask("[cyan]CF-Access-Client-Id[/cyan]")
            console.print("[cyan]CF-Access-Client-Secret[/cyan]: ", end="")
            self.cf_client_secret = getpass.getpass("")
            console.print()

        if not self.cf_client_id or not self.cf_client_secret:
            console.print("[red]Cloudflare credentials required[/red]")
            return False

        self.cf_client = CloudflareAccessClient(self.cf_client_id, self.cf_client_secret)
        self._initialized = True
        return True

    def build_sensor_url(self, sensor_connection: Dict) -> str:
        """
        Build sensor URL from Portal's sensorRemoteConnection data.

        Args:
            sensor_connection: Dict with 'host', 'port', 'https' keys

        Returns:
            Full sensor URL
        """
        host = sensor_connection.get('host', '')
        port = sensor_connection.get('port', 443)
        use_https = sensor_connection.get('https', True)

        protocol = 'https' if use_https else 'http'

        if port and port not in (80, 443):
            return f"{protocol}://{host}:{port}"
        return f"{protocol}://{host}"

    def get_existing_snapshot(self, sensor_url: str, snapshot_uuid: str) -> Optional[Dict]:
        """
        Get an existing snapshot from sensor by its UUID (from Portal's sensorSnapshotId).

        Args:
            sensor_url: Full sensor URL
            snapshot_uuid: The sensorSnapshotId from Portal validation data

        Returns:
            Snapshot data dict with 'spaces' containing images, or None on failure
        """
        # Check in-memory cache first
        cache_key = (sensor_url, snapshot_uuid)
        if cache_key in self._snapshot_cache:
            return self._snapshot_cache[cache_key]

        # Extract sensor name from URL for cache filename
        # e.g., "https://sensor-name.example.com" -> "sensor-name.example.com"
        from urllib.parse import urlparse
        parsed = urlparse(sensor_url)
        sensor_name = parsed.netloc.replace(':', '_')  # Replace port colon if present

        # Check disk cache
        cache_file = self._cache_dir / f"{snapshot_uuid}_{sensor_name}.json"
        if cache_file.exists():
            try:
                import json
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                # Add to in-memory cache
                self._snapshot_cache[cache_key] = data
                return data
            except Exception:
                # Cache corrupted, delete and re-fetch
                cache_file.unlink()

        endpoint = f"{sensor_url}/plugin/snapshot/get/{snapshot_uuid}"

        try:
            response = self.cf_client.get(endpoint)

            if 'response' in response and 'data' in response['response']:
                data = response['response']['data']
                # Save to disk cache
                try:
                    import json
                    with open(cache_file, 'w') as f:
                        json.dump(data, f)
                except Exception:
                    pass  # Don't fail if caching fails
                # Add to in-memory cache
                self._snapshot_cache[cache_key] = data
                return data
            return None

        except Exception as e:
            error_msg = str(e)
            if 'Authentication failed' in error_msg or 'Cloudflare' in error_msg:
                print(f"[ERROR] Authentication failed: {e}")
            else:
                print(f"[ERROR] Failed to get snapshot {snapshot_uuid}: {e}")
            return None

    def get_space_image(self, sensor_connection: Dict, snapshot_uuid: str,
                        space_name: str, bounds: List[Dict] = None) -> Optional[np.ndarray]:
        """
        Get a specific space's image from an existing sensor snapshot.

        Args:
            sensor_connection: Portal's sensorRemoteConnection data
            snapshot_uuid: The sensorSnapshotId from Portal validation data
            space_name: Name of the parking space
            bounds: Optional ROI bounds for 'extract' mode (list of 4 points with 'x', 'y' keys)

        Returns:
            BGR numpy array of the space image, or None if not found
        """
        sensor_url = self.build_sensor_url(sensor_connection)
        snapshot_data = self.get_existing_snapshot(sensor_url, snapshot_uuid)

        if not snapshot_data:
            return None

        spaces = snapshot_data.get('spaces', [])

        for space in spaces:
            if space.get('name') == space_name:
                import base64

                # Mode: extract from full image
                if self.roi_mode in ('extract', 'extract_jpeg', 'extract_png'):
                    # Get the full sensor image and extract ROI ourselves
                    # Choose source based on mode
                    if self.roi_mode == 'extract_png':
                        # Try PNG full image first
                        full_image_b64 = snapshot_data.get('fullPngImage') or snapshot_data.get('pngFullImage')
                        if not full_image_b64:
                            print(f"[WARN] No PNG full image for {space_name}, falling back to JPEG")
                            full_image_b64 = snapshot_data.get('image') or snapshot_data.get('fullImage')
                    else:
                        # extract or extract_jpeg: use JPEG full image
                        full_image_b64 = snapshot_data.get('image') or snapshot_data.get('fullImage')

                    space_bounds = space.get('roi') or bounds  # Try space's ROI or passed bounds

                    if full_image_b64 and space_bounds:
                        try:
                            img_data = base64.b64decode(full_image_b64)
                            img_array = np.frombuffer(img_data, dtype=np.uint8)
                            full_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                            if full_img is not None:
                                # Extract ROI using our extraction code
                                # Convert to PIL for TFLiteInference.extract_roi compatibility
                                from PIL import Image
                                full_img_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
                                pil_img = Image.fromarray(full_img_rgb)

                                # Extract ROI (returns RGB numpy array)
                                roi_rgb = TFLiteInference.extract_roi(
                                    pil_img, space_bounds,
                                    use_perspective_warp=True,  # Match sensor behavior
                                    use_max_dims=True,
                                    space_name=space_name
                                )

                                # Convert back to BGR
                                return cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
                        except Exception as e:
                            print(f"[ERROR] Failed to extract ROI for {space_name}: {e}")
                            # Fall through to use pngImage/image

                # Mode: png (prefer pngImage) or jpeg (prefer image)
                # Also serves as fallback when extraction fails
                if self.roi_mode in ('jpeg', 'extract_jpeg'):
                    image_b64 = space.get('image') or space.get('pngImage', '')
                else:  # 'png', 'extract', or 'extract_png' fallback
                    image_b64 = space.get('pngImage') or space.get('image', '')

                if not image_b64:
                    return None

                try:
                    img_data = base64.b64decode(image_b64)
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    return img  # BGR format
                except Exception as e:
                    print(f"[ERROR] Failed to decode image for {space_name}: {e}")
                    return None

        return None


class PortalAPIClient:
    """Client for ECO Parking Portal GraphQL API"""

    def __init__(self, refresh_token: str):
        self.refresh_token = refresh_token
        self.access_token = None
        self.token_expiry = None
        self.graphql_url = 'https://api.ecoparkingtechnologies.com/graphql'
        self.auth_url = 'https://identity.ecoparkingtechnologies.com/token'
        self.client_id = 'EcoParking.ClientApi'

    def get_access_token(self) -> str:
        """Exchange refresh token for access token using OAuth2"""
        if self.access_token and self.token_expiry:
            if datetime.now() < self.token_expiry - timedelta(minutes=5):
                return self.access_token

        payload = {
            'grant_type': 'refresh_token',
            'refresh_token': self.refresh_token,
            'client_id': self.client_id
        }

        try:
            response = requests.post(self.auth_url, data=payload)
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data['access_token']
            expires_in = token_data.get('expires_in', 3600)
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)

            return self.access_token

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get access token: {e}")

    def execute_query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """Execute GraphQL query with automatic token refresh"""
        access_token = self.get_access_token()

        headers = {
            'Authorization': f'Bearer {access_token}',
            'Content-Type': 'application/json'
        }

        payload = {'query': query}
        if variables:
            payload['variables'] = variables

        try:
            response = requests.post(self.graphql_url, json=payload, headers=headers)

            # Try to get response body for debugging
            try:
                result = response.json()
            except:
                result = {'error': response.text}

            response.raise_for_status()

            if 'errors' in result:
                error_details = json.dumps(result['errors'], indent=2)
                raise Exception(f"GraphQL errors:\n{error_details}")

            return result.get('data')

        except requests.exceptions.HTTPError as e:
            # Include response body in error message
            error_msg = f"Query failed: {e}\n"
            if 'errors' in result:
                error_msg += f"GraphQL errors:\n{json.dumps(result['errors'], indent=2)}"
            elif 'error' in result:
                error_msg += f"Response:\n{result['error'][:500]}"
            raise Exception(error_msg)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Query failed: {e}")

    def get_organizations(self) -> List[Dict]:
        """Get all organizations"""
        query = """
        query GetOrganizations {
          organizations(condition: {isDeleted: false}, orderBy: NAME_ASC) {
            nodes {
              id
              name
              displayName
              description
              __typename
            }
            __typename
          }
        }
        """
        result = self.execute_query(query)
        if result and 'organizations' in result:
            return result['organizations']['nodes']
        return []

    def get_sites_for_org(self, org_id: int) -> List[Dict]:
        """Get all sites for an organization"""
        query = """
        query GetSites($id: Int!) {
          sites(condition: {organizationId: $id, isDeleted: false}) {
            nodes {
              displayName
              id
              timeZoneName
              latestSiteParkingUsages {
                nodes {
                  availableCount
                  occupiedCount
                  totalCount
                  utilizedPercent
                  timestamp
                  __typename
                }
                __typename
              }
              __typename
            }
            __typename
          }
        }
        """
        variables = {'id': org_id}
        result = self.execute_query(query, variables)
        if result and 'sites' in result:
            return result['sites']['nodes']
        return []

    def get_validations_for_site(self, site_id: int) -> List[Dict]:
        """Get all validations for a site"""
        query = """
        query GetValidations($siteId: Int!) {
          validations(condition: {siteId: $siteId}) {
            nodes {
              id
              createdTimestamp
              validationState
              snapshot {
                runAt
                id
                __typename
              }
              parkingSpaceAccuracy
              __typename
            }
            __typename
          }
        }
        """
        variables = {'siteId': site_id}
        result = self.execute_query(query, variables)
        if result and 'validations' in result:
            return result['validations']['nodes']
        return []

    def get_validation_details(self, validation_id: str) -> Optional[Dict]:
        """Get detailed validation data including parking spaces and responses"""
        query = """
        query GetValidation($id: UUID!) {
          validation(id: $id) {
            validationParkingSpaces {
              nodes {
                id
                snapshotParkingSpace {
                  id
                  reportedOccupancyStatus
                  imageUrl
                  imageUrlSigned
                  originalImageUrlSigned
                  bounds
                  stable
                  thresholdLow
                  thresholdHigh
                  rawInference
                  inference
                  filteredInference
                  sensorSnapshotId
                  parkingSpace {
                    id
                    name
                    __typename
                  }
                  parkingAttribute {
                    id
                    name
                    displayName
                    ledRgbaValue
                    priority
                    __typename
                  }
                  indicatedBySensor {
                    id
                    configurationName
                    configurationDescription
                    __typename
                  }
                  detectedBySensor {
                    id
                    configurationName
                    configurationDescription
                    sensorRemoteConnection {
                      host
                      port
                      https
                      __typename
                    }
                    __typename
                  }
                  __typename
                }
                validationParkingSpaceResponses {
                  nodes {
                    occupancyStatus
                    workerId
                    __typename
                  }
                  __typename
                }
                validationParkingSpaceResponseOverride {
                  occupancyStatus
                  user {
                    familyName
                    givenName
                    id
                    __typename
                  }
                  __typename
                }
                __typename
              }
              __typename
            }
            snapshot {
              createdTimestamp
              id
              __typename
            }
            createdTimestamp
            validationState
            parkingSpaceAccuracy
            __typename
          }
        }
        """
        variables = {'id': validation_id}
        result = self.execute_query(query, variables)
        if result and 'validation' in result:
            return result['validation']
        return None


