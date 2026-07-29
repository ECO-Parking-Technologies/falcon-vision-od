# Falcon Vision override of cvat.settings.production (bind-mounted by compose).
# Adds CSRF_TRUSTED_ORIGINS support — stock CVAT production settings omit it,
# which breaks all POSTs arriving through an https reverse proxy/tunnel.
from cvat.settings.base import *  # noqa: F401,F403
import os

DEBUG = False

NUCLIO['HOST'] = os.getenv('CVAT_NUCLIO_HOST', 'nuclio')  # noqa: F405

SENDFILE_BACKEND = 'django_sendfile.backends.nginx'
SENDFILE_URL = '/'

_csrf = os.getenv('CSRF_TRUSTED_ORIGINS', '')
if _csrf:
    CSRF_TRUSTED_ORIGINS = [o.strip() for o in _csrf.split(',') if o.strip()]
