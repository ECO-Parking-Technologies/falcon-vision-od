# Falcon Vision - New Annotation Pipeline Setup (CVAT)

## Clone CVAT

```bash
git clone https://github.com/openvinotoolkit/cvat.git
cd cvat
git checkout v2.23.1
```

## Update `docker-compose.yml`

Apply the following changes:

```diff
diff --git a/docker-compose.yml b/docker-compose.yml
index b00ff6ba3..94429fdcd 100644
--- a/docker-compose.yml
+++ b/docker-compose.yml
@@ -106,6 +106,7 @@ services:
        - cvat_data:/home/django/data
        - cvat_keys:/home/django/keys
        - cvat_logs:/home/django/logs
+      - cvat_share:/home/django/share:ro
    networks:
        cvat:
        aliases:
@@ -141,6 +142,7 @@ services:
        - cvat_data:/home/django/data
        - cvat_keys:/home/django/keys
        - cvat_logs:/home/django/logs
+      - cvat_share:/home/django/share:ro
    networks:
        - cvat
@@ -157,6 +159,7 @@ services:
        - cvat_data:/home/django/data
        - cvat_keys:/home/django/keys
        - cvat_logs:/home/django/logs
+      - cvat_share:/home/django/share:ro
    networks:
        - cvat
@@ -173,6 +176,7 @@ services:
        - cvat_data:/home/django/data
        - cvat_keys:/home/django/keys
        - cvat_logs:/home/django/logs
+      - cvat_share:/home/django/share:ro
    networks:
        - cvat
@@ -237,6 +241,7 @@ services:
        - cvat_data:/home/django/data
        - cvat_keys:/home/django/keys
        - cvat_logs:/home/django/logs
+      - cvat_share:/home/django/share:ro
    networks:
        - cvat
@@ -259,7 +264,7 @@ services:
    container_name: traefik
    restart: always
    ports:
-      - 8080:8080
+      - 8085:8080
        - 8090:8090
    environment:
        CVAT_HOST: ${CVAT_HOST:-localhost}
@@ -407,6 +412,11 @@ volumes:
    cvat_inmem_db:
    cvat_events_db:
    cvat_cache_db:
+  cvat_share:
+    driver_opts:
+      type: none
+      device: /media/lopezemi/Expansion/falcon-vision-ml/artifacts/data_pipeline
+      o: bind
```

> **Note:**  
> - Replace `/media/lopezemi/Expansion/falcon-vision-ml/artifacts/data_pipeline` with the **path on your machine** where your **garage training images** are stored.
> - This path must be **accessible to Docker** and readable by CVAT.
> - `cvat_share` is mounted **read-only** (`:ro`) inside the container at `/home/django/share`.

---

## Start CVAT

```bash
export CVAT_HOST=<your_host_ipv4_address>
export CVAT_VERSION=v2.23.1
docker compose up -d
```

> **Example:**  
> If your host machine IP is `192.168.1.30`, then set:
> ```bash
> export CVAT_HOST=192.168.1.30
> ```

---

# CVAT Annotation Guidelines for Vehicle Detection

## Setup Instructions

1. **Access the CVAT Interface**:
   - Navigate to `http://<your_host_ipv4_address>:8085`
   - Create an account and log in.
   - Create an **Organization** and switch to it.

2. **Project Creation**:
   - Create a new project named `Falcon Vision`.
   - Use default settings.
   - Upload project labels manually (vehicle, person, etc).

3. **Task Creation**:
   - Name the task based on **garage + sensor**, e.g., `arlington-fv1b999e`.
   - Assign it to the `Falcon Vision` project.
   - **Labels**: come from the project.
   - **Files**:
     - Navigate to the garage/sensor folder (e.g., `arlington/training_images/fv1b999e`).
     - Select all `.png` images.
   - Click **Submit & Open** (image extraction may take time).

4. **Task Management**:
   - Open the task after creation.
   - Set:
     - **Assignee** (the person working on it)
     - **Stage**: `annotation`
     - **State**: `in progress`

---

## Running Pre-Annotation (EfficientDet)

1. Activate your virtual environment:

   ```bash
   source falcon-vision-od-venv/bin/activate
   ```

2. Run pre-annotation:

   ```bash
   python3 preannotation/run_preannotation.py --config preannotation/config.yaml --visualize 3
   ```

3. After processing, `.coco.json` files are generated inside each sensor folder.

4. **Import Pre-Annotations into CVAT**:
   - Open the matching CVAT task.
   - Go to **Menu → Upload annotations**.
   - Select format: **COCO 1.0**.
   - Upload the generated `annotations.coco.json`.

---

## Annotation Guidelines

### General Principles
- Focus on **vehicles present** in the scene.
- Maintain clear and consistent annotations.

### When to Annotate
- Label vehicles:
  - Clearly visible
  - In priority detection zones
- Ignore:
  - Highly occluded or distant vehicles

### Bounding Box Rules
- Draw **tight boxes** around vehicles.
- Do not annotate vehicles that are mostly hidden or irrelevant.

### Specific Scenarios
- **Overlapping Cars**: Annotate only the most visible car.
- **Far-Back Vehicles**: Skip tiny or blurry cars.

### Tools and Shortcuts
- **Draw Rectangle Tool** with label = `vehicle`.
- Useful shortcuts:
  - `F`: Next image
  - `D`: Previous image
  - `N`: Draw Rectangle
  - `O`: Mark as outside
  - `ESC`: Cancel

### Tracking
- Use **Track** to propagate boxes across frames when the vehicle remains visible.
- Adjust bounding boxes as vehicles move.

---

## Annotation Review

- Review all annotations before export.
- Check:
  - Tight boxes
  - Consistent labeling
  - No irrelevant objects labeled

---

## Training

Training will be based on **EfficientDet D4**, using COCO-format annotations.
