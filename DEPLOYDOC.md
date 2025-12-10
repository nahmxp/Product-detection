# 📋 COMPLETE DOCUMENTATION: Product Detection Docker & Azure GPU Deployment

## 🎯 PROJECT OVERVIEW

**What This Project Does:**
- **Product Detection System** using YOLO11 (Ultralytics) for object detection and segmentation
- **Streamlit Web App** (`app.py`) - Full training dashboard with:
  - COCO to YOLO dataset conversion
  - Data augmentation (Albumentations)
  - Annotation checking
  - Model training (with GPU support)
  - Continual learning
  - TFLite conversion
  - Inference/Detection
- **Camera App** (`camera.py`) - Real-time detection via webcam
- **Training Scripts** (`train.py`) - Standalone YOLO training
- **TFLite Tools** - Convert and test TensorFlow Lite models

**Tech Stack:**
- Python 3.12
- PyTorch (with CUDA for GPU)
- Ultralytics YOLO11
- Streamlit
- OpenCV, Albumentations, TensorFlow

---

## 🐳 PART 1: DOCKERFILE ANALYSIS

### Current Dockerfile Issues for GPU Training:
```dockerfile
FROM python:3.12-slim  # ❌ This is CPU-only base image
# Missing: CUDA toolkit, cuDNN, GPU drivers
# Missing: PyTorch with GPU support
```

### What You Need to Change:

**For GPU Support, you need a DIFFERENT base image:**

1. **Option A: NVIDIA CUDA Base (Recommended for Training)**
   ```dockerfile
   FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
   # or
   FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
   ```

2. **Option B: PyTorch Official GPU Image**
   ```dockerfile
   FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
   ```

---

## 🔧 PART 2: CREATING GPU-ENABLED DOCKER CONTAINER

### Step 1: Update Your Dockerfile

Create a new file called `Dockerfile.gpu`:

```dockerfile
# ===== GPU-ENABLED DOCKERFILE =====
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    PORT=8501 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    git \
    ffmpeg \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /app

# Copy requirements and install Python deps
COPY requirements.txt /app/requirements.txt

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir onnx2tf

# Copy project files
COPY . /app

# Create necessary directories
RUN mkdir -p /app/Model /app/dataset /app/runs

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Step 2: Build the Docker Image

```bash
# Navigate to your project directory
cd "/media/xpert-ai/Documents/NDEV/Product detection"

# Build the GPU-enabled image
docker build -f Dockerfile.gpu -t product-detection-gpu:latest .

# Check the image was created
docker images | grep product-detection
```

### Step 3: Run the Container Locally (with GPU)

**Prerequisites:**
- NVIDIA Docker runtime installed (`nvidia-docker2`)
- NVIDIA GPU drivers installed on host

```bash
# Install NVIDIA Container Toolkit (if not installed)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Run your container with GPU access
docker run -d \
    --name product-detection \
    --gpus all \
    -p 8501:8501 \
    -v "$(pwd)/dataset:/app/dataset" \
    -v "$(pwd)/Model:/app/Model" \
    -v "$(pwd)/runs:/app/runs" \
    --restart unless-stopped \
    product-detection-gpu:latest

# Check container logs
docker logs -f product-detection

# Access the app
# Open browser: http://localhost:8501
```

### Step 4: Verify GPU is Available Inside Container

```bash
# Enter the running container
docker exec -it product-detection bash

# Inside container, run Python check
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Exit container
exit
```

---

## ☁️ PART 3: DEPLOYING TO AZURE WITH GPU (Azure Student Account)

### Prerequisites:
- Azure Student Account (comes with $100 credit)
- Azure CLI installed locally
- Docker image ready

### Architecture Options:

#### **Option A: Azure Container Instances (ACI) with GPU** ⚠️ Limited GPU Support
- ❌ **NOT RECOMMENDED** - ACI doesn't support GPUs well
- Better for CPU workloads

#### **Option B: Azure Virtual Machine with GPU** ✅ **RECOMMENDED**
- Full control, persistent storage
- NC-series VMs (NVIDIA Tesla)
- Best for training workloads

#### **Option C: Azure Container Apps** ⚠️ No GPU Support
- ❌ **NOT SUITABLE** - No GPU

#### **Option D: Azure Kubernetes Service (AKS)** ⭐ **BEST for Production**
- Scalable, production-ready
- Supports GPU node pools
- More complex setup

---

## 🚀 RECOMMENDED: DEPLOY ON AZURE VM WITH GPU

### Step 1: Install Azure CLI (Local Machine)

```bash
# Install Azure CLI on Ubuntu/Debian
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login

# Set your subscription (if you have multiple)
az account list --output table
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

### Step 2: Check Available GPU VMs in Your Region

```bash
# List available GPU VM sizes (NC-series for NVIDIA)
az vm list-sizes --location eastus --output table | grep NC

# Common GPU VMs for students:
# - Standard_NC6 (1x NVIDIA Tesla K80, 6 vCPUs, 56GB RAM) - ~$0.90/hour
# - Standard_NC6s_v3 (1x NVIDIA V100, 6 vCPUs, 112GB RAM) - ~$3.06/hour
# - Standard_NC4as_T4_v3 (1x NVIDIA T4, 4 vCPUs, 28GB RAM) - ~$0.526/hour ✅ BEST VALUE

# Check quota
az vm list-usage --location eastus --output table | grep NC
```

### Step 3: Create Resource Group

```bash
# Create a resource group
az group create \
    --name product-detection-rg \
    --location eastus

# Verify
az group show --name product-detection-rg
```

### Step 4: Create GPU-Enabled Virtual Machine

```bash
# Create Ubuntu VM with GPU
az vm create \
    --resource-group product-detection-rg \
    --name product-detection-vm \
    --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
    --size Standard_NC4as_T4_v3 \
    --admin-username azureuser \
    --generate-ssh-keys \
    --public-ip-sku Standard \
    --priority Spot \
    --max-price -1 \
    --eviction-policy Deallocate

# Note: Using --priority Spot makes it cheaper but can be evicted
# Remove --priority Spot --max-price -1 --eviction-policy Deallocate for guaranteed VM

# Open port 8501 for Streamlit
az vm open-port \
    --resource-group product-detection-rg \
    --name product-detection-vm \
    --port 8501 \
    --priority 1001

# Get public IP
az vm show \
    --resource-group product-detection-rg \
    --name product-detection-vm \
    --show-details \
    --query publicIps \
    --output tsv
```

### Step 5: SSH Into VM and Setup Environment

```bash
# SSH into the VM (replace with your public IP)
ssh azureuser@YOUR_VM_PUBLIC_IP

# ===== INSIDE THE VM =====

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA drivers and CUDA
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify GPU
nvidia-smi

# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Step 6: Push Docker Image to Azure Container Registry (ACR)

**On Local Machine:**

```bash
# Create Azure Container Registry
az acr create \
    --resource-group product-detection-rg \
    --name productdetectionacr \
    --sku Basic \
    --admin-enabled true

# Get ACR login credentials
az acr credential show \
    --name productdetectionacr \
    --resource-group product-detection-rg

# Login to ACR from local Docker
az acr login --name productdetectionacr

# Tag your local image
docker tag product-detection-gpu:latest productdetectionacr.azurecr.io/product-detection-gpu:latest

# Push to ACR
docker push productdetectionacr.azurecr.io/product-detection-gpu:latest

# Verify
az acr repository list --name productdetectionacr --output table
```

### Step 7: Pull and Run Container on Azure VM

**On Azure VM (SSH session):**

```bash
# Login to ACR from VM
docker login productdetectionacr.azurecr.io \
    --username productdetectionacr \
    --password "YOUR_ACR_PASSWORD"

# Pull the image
docker pull productdetectionacr.azurecr.io/product-detection-gpu:latest

# Create directories for persistent data
mkdir -p ~/product-detection/dataset
mkdir -p ~/product-detection/Model
mkdir -p ~/product-detection/runs

# Run the container with GPU
docker run -d \
    --name product-detection \
    --gpus all \
    -p 8501:8501 \
    -v ~/product-detection/dataset:/app/dataset \
    -v ~/product-detection/Model:/app/Model \
    -v ~/product-detection/runs:/app/runs \
    --restart unless-stopped \
    productdetectionacr.azurecr.io/product-detection-gpu:latest

# Check logs
docker logs -f product-detection

# Verify GPU inside container
docker exec -it product-detection python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 8: Access Your Application

```bash
# Get public IP (if you forgot)
az vm show \
    --resource-group product-detection-rg \
    --name product-detection-vm \
    --show-details \
    --query publicIps \
    --output tsv

# Open in browser:
# http://YOUR_VM_PUBLIC_IP:8501
```

---

## 📦 PART 4: TRANSFERRING DATA TO AZURE VM

### Option A: Use SCP (Secure Copy)

```bash
# From local machine, upload dataset
scp -r "/media/xpert-ai/Documents/NDEV/Product detection/dataset" \
    azureuser@YOUR_VM_PUBLIC_IP:~/product-detection/

# Upload models
scp -r "/media/xpert-ai/Documents/NDEV/Product detection/Model" \
    azureuser@YOUR_VM_PUBLIC_IP:~/product-detection/
```

### Option B: Use Azure Storage (Better for Large Datasets)

```bash
# Create storage account
az storage account create \
    --name productdetectionstorage \
    --resource-group product-detection-rg \
    --location eastus \
    --sku Standard_LRS

# Create blob container
az storage container create \
    --name datasets \
    --account-name productdetectionstorage \
    --public-access off

# Upload data (install Azure Storage Explorer GUI or use CLI)
# Install azcopy
wget https://aka.ms/downloadazcopy-v10-linux
tar -xvf downloadazcopy-v10-linux
sudo cp ./azcopy_linux_amd64_*/azcopy /usr/local/bin/

# Get storage key
az storage account keys list \
    --resource-group product-detection-rg \
    --account-name productdetectionstorage

# Upload dataset
azcopy copy "/media/xpert-ai/Documents/NDEV/Product detection/dataset" \
    "https://productdetectionstorage.blob.core.windows.net/datasets" \
    --recursive=true
```

---

## 💰 PART 5: COST OPTIMIZATION

### GPU VM Costs (Approximate):
- **Standard_NC4as_T4_v3**: ~$0.526/hour (~$380/month)
- **Spot Instance**: 70-90% discount (can be evicted)

### Money-Saving Tips:

1. **Use Spot Instances** (already in command above)
   ```bash
   --priority Spot --max-price -1
   ```

2. **Auto-Shutdown When Idle**
   ```bash
   # Set auto-shutdown at 11 PM daily
   az vm auto-shutdown \
       --resource-group product-detection-rg \
       --name product-detection-vm \
       --time 2300
   ```

3. **Deallocate VM When Not Training**
   ```bash
   # Stop VM (releases compute, keeps storage)
   az vm deallocate --resource-group product-detection-rg --name product-detection-vm
   
   # Start when needed
   az vm start --resource-group product-detection-rg --name product-detection-vm
   ```

4. **Monitor Spending**
   ```bash
   # Set budget alert
   az consumption budget create \
       --budget-name monthly-budget \
       --amount 100 \
       --time-grain Monthly \
       --start-date 2025-12-01 \
       --end-date 2026-12-01
   ```

---

## 🔒 PART 6: SECURITY BEST PRACTICES

### 1. Restrict Network Access
```bash
# Create Network Security Group rule for your IP only
MY_IP=$(curl -s ifconfig.me)

az vm open-port \
    --resource-group product-detection-rg \
    --name product-detection-vm \
    --port 8501 \
    --priority 1001

# Restrict to your IP
az network nsg rule update \
    --resource-group product-detection-rg \
    --nsg-name product-detection-vmNSG \
    --name open-port-8501 \
    --source-address-prefixes $MY_IP/32
```

### 2. Use HTTPS (Add Nginx Reverse Proxy)
```bash
# Inside VM, install Nginx
sudo apt-get install -y nginx certbot python3-certbot-nginx

# Configure Nginx as reverse proxy
sudo nano /etc/nginx/sites-available/product-detection

# Add this configuration:
server {
    listen 80;
    server_name YOUR_DOMAIN_OR_IP;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/product-detection /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Open port 80
az vm open-port --port 80 --resource-group product-detection-rg --name product-detection-vm
```

---

## 📝 PART 7: QUICK REFERENCE COMMANDS

### Local Development:
```bash
# Build image
docker build -f Dockerfile.gpu -t product-detection-gpu:latest .

# Run locally
docker run -d --name product-detection --gpus all -p 8501:8501 \
    -v "$(pwd)/dataset:/app/dataset" \
    -v "$(pwd)/Model:/app/Model" \
    -v "$(pwd)/runs:/app/runs" \
    product-detection-gpu:latest

# Stop
docker stop product-detection && docker rm product-detection
```

### Azure Management:
```bash
# Start VM
az vm start --resource-group product-detection-rg --name product-detection-vm

# Stop VM (deallocate to save money)
az vm deallocate --resource-group product-detection-rg --name product-detection-vm

# Get VM status
az vm get-instance-view --resource-group product-detection-rg --name product-detection-vm \
    --query instanceView.statuses[1] --output table

# Delete everything (cleanup)
az group delete --name product-detection-rg --yes --no-wait
```

### Monitoring:
```bash
# SSH into VM
ssh azureuser@YOUR_VM_IP

# Inside VM - check GPU usage
watch -n 1 nvidia-smi

# Check Docker logs
docker logs -f product-detection

# Check disk space
df -h

# Check container resource usage
docker stats product-detection
```

---

## 🎓 SUMMARY FOR FUTURE REFERENCE

### What You Need:

1. **Modified Dockerfile (Dockerfile.gpu)** - Uses NVIDIA CUDA base image
2. **Azure Student Account** - $100 credit
3. **Azure VM with GPU** - NC4as_T4_v3 (cheapest GPU option)
4. **Azure Container Registry** - To store your Docker images
5. **NVIDIA Docker Runtime** - On both local and Azure VM

### Complete Workflow:

```
Local Machine:
1. Create Dockerfile.gpu
2. Build image: docker build -f Dockerfile.gpu -t product-detection-gpu:latest .
3. Test locally: docker run --gpus all -p 8501:8501 product-detection-gpu:latest

Push to Azure:
4. Create ACR: az acr create
5. Push image: docker push productdetectionacr.azurecr.io/product-detection-gpu:latest

Azure VM:
6. Create GPU VM: az vm create --size Standard_NC4as_T4_v3
7. Install NVIDIA drivers + Docker
8. Pull image from ACR
9. Run container: docker run --gpus all -p 8501:8501 <image>
10. Access: http://YOUR_VM_IP:8501

Training:
11. Upload datasets via SCP or Azure Storage
12. Train models through Streamlit UI
13. Download trained models back to local
```

### Costs to Watch:
- **VM Runtime**: ~$0.53/hour (Spot) or ~$380/month (regular)
- **Storage**: ~$0.05/GB/month
- **ACR**: ~$5/month (Basic tier)
- **Data Transfer**: First 100GB free/month

**Total Estimated**: $10-50/month if you shut down VM when not training

---

## 🔧 TROUBLESHOOTING GUIDE

### Problem: GPU Not Detected in Container

**Solution 1: Check NVIDIA Docker Runtime**
```bash
# Verify nvidia-docker2 is installed
dpkg -l | grep nvidia-docker

# If not installed, install it
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Solution 2: Check Docker Daemon Configuration**
```bash
# Edit Docker daemon config
sudo nano /etc/docker/daemon.json

# Should contain:
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}

# Restart Docker
sudo systemctl restart docker
```

**Solution 3: Verify GPU Drivers on Host**
```bash
# Check NVIDIA driver
nvidia-smi

# If command not found, install drivers
sudo ubuntu-drivers autoinstall
sudo reboot
```

---

### Problem: Out of Memory During Training

**Solution 1: Reduce Batch Size**
Edit your training configuration in `app.py` or `train.py`:
```python
batch=8,  # Reduce from 24 to 8 or 4
```

**Solution 2: Reduce Image Size**
```python
imgsz=416,  # Reduce from 640 to 416
```

**Solution 3: Enable Mixed Precision Training**
```python
amp=True,  # Automatic Mixed Precision
```

**Solution 4: Monitor GPU Memory**
```bash
# Inside container or VM
watch -n 1 nvidia-smi
```

---

### Problem: Azure VM Creation Fails (Quota Exceeded)

**Solution: Request Quota Increase**
```bash
# Check current quota
az vm list-usage --location eastus --output table | grep StandardNCFamily

# Request quota increase via Azure Portal:
# 1. Go to Azure Portal > Subscriptions
# 2. Click on your subscription
# 3. Select "Usage + quotas"
# 4. Search for "NC" family
# 5. Click "Request increase"
# 6. Fill form (usually approved within 24 hours for students)
```

**Alternative: Try Different Region**
```bash
# Check VM availability in other regions
az vm list-sizes --location westus2 --output table | grep NC
az vm list-sizes --location centralus --output table | grep NC
```

---

### Problem: Docker Build Fails on Large Files

**Solution: Use .dockerignore**
Create `.dockerignore` file:
```
dataset/
Model/
runs/
*.pt
*.pth
*.onnx
*.tflite
__pycache__/
*.pyc
.git/
.vscode/
venv/
```

---

### Problem: Slow Data Transfer to Azure VM

**Solution: Use Azure Storage Sync**
```bash
# Install Azure File Sync agent on VM
wget https://aka.ms/afs/agent/Linux
sudo dpkg -i storagesyncagent_*_amd64.deb

# Or use rsync with compression
rsync -avz --progress "/media/xpert-ai/Documents/NDEV/Product detection/dataset" \
    azureuser@YOUR_VM_IP:~/product-detection/
```

---

### Problem: Streamlit App Not Accessible from Browser

**Solution 1: Check Firewall Rules**
```bash
# Verify NSG rules
az network nsg rule list \
    --resource-group product-detection-rg \
    --nsg-name product-detection-vmNSG \
    --output table

# Verify port is open
sudo ufw status
sudo ufw allow 8501/tcp
```

**Solution 2: Check Container Logs**
```bash
docker logs product-detection

# If container crashed, restart it
docker restart product-detection
```

**Solution 3: Verify Streamlit Configuration**
```bash
# Inside container
docker exec -it product-detection bash
streamlit config show
```

---

### Problem: Model Training Crashes Silently

**Solution: Enable Verbose Logging**
```python
# In your training script
import logging
logging.basicConfig(level=logging.DEBUG)

# Or run with verbose flag
results = model.train(
    verbose=True,
    # ... other parameters
)
```

**Check System Resources:**
```bash
# Monitor during training
htop  # CPU and RAM
nvidia-smi -l 1  # GPU every 1 second
df -h  # Disk space
```

---

### Problem: ACR Push/Pull Authentication Fails

**Solution 1: Regenerate ACR Credentials**
```bash
# Get new credentials
az acr credential renew \
    --name productdetectionacr \
    --password-name password

# Show credentials
az acr credential show --name productdetectionacr
```

**Solution 2: Use Service Principal (More Secure)**
```bash
# Create service principal
az ad sp create-for-rbac \
    --name productdetectionacr-sp \
    --scopes $(az acr show --name productdetectionacr --query id --output tsv) \
    --role acrpull

# Login with service principal
docker login productdetectionacr.azurecr.io \
    --username <appId> \
    --password <password>
```

---

### Problem: High Azure Costs

**Solution: Set Up Cost Alerts and Automation**

**1. Create Auto-Shutdown Script:**
```bash
# Create shutdown script on VM
sudo nano /usr/local/bin/auto-shutdown-idle.sh

# Add:
#!/bin/bash
IDLE_TIME=3600  # 1 hour in seconds
CPU_THRESHOLD=5  # 5% CPU usage

CURRENT_CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
if (( $(echo "$CURRENT_CPU < $CPU_THRESHOLD" | bc -l) )); then
    echo "CPU below threshold. Shutting down..."
    sudo shutdown -h now
fi

# Make executable
sudo chmod +x /usr/local/bin/auto-shutdown-idle.sh

# Add to cron (check every 30 minutes)
sudo crontab -e
# Add: */30 * * * * /usr/local/bin/auto-shutdown-idle.sh
```

**2. Use Azure Cost Management:**
```bash
# Create budget
az consumption budget create \
    --budget-name student-monthly \
    --amount 100 \
    --category cost \
    --time-grain Monthly \
    --time-period start=$(date +%Y-%m-01) end=2026-12-31 \
    --resource-group product-detection-rg
```

**3. Schedule VM Deallocate:**
```bash
# Install Azure CLI on VM
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Create shutdown script
cat > ~/deallocate-at-night.sh << 'EOF'
#!/bin/bash
az vm deallocate --resource-group product-detection-rg --name product-detection-vm
EOF

chmod +x ~/deallocate-at-night.sh

# Schedule with cron (11 PM daily)
crontab -e
# Add: 0 23 * * * /home/azureuser/deallocate-at-night.sh
```

---

## 🎯 ADVANCED OPTIMIZATIONS

### 1. Multi-Stage Docker Build (Reduce Image Size)

Create `Dockerfile.multistage`:
```dockerfile
# Stage 1: Builder
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip \
    build-essential git wget

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --user \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    PORT=8501 \
    PATH=/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    ffmpeg \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the dependencies from builder
COPY --from=builder /root/.local /root/.local

COPY . /app

RUN mkdir -p /app/Model /app/dataset /app/runs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Enable Docker BuildKit (Faster Builds)

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build with cache
docker build -f Dockerfile.multistage \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    -t product-detection-gpu:latest .
```

### 3. Use Azure Spot VMs for Training (Save 70-90%)

```bash
# Create Spot VM with custom eviction notice
az vm create \
    --resource-group product-detection-rg \
    --name product-detection-spot-vm \
    --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
    --size Standard_NC4as_T4_v3 \
    --admin-username azureuser \
    --generate-ssh-keys \
    --priority Spot \
    --max-price 0.3 \
    --eviction-policy Deallocate

# Create script to handle eviction
cat > ~/handle-eviction.sh << 'EOF'
#!/bin/bash
# Save model checkpoint before eviction
docker exec product-detection python3 -c "
from ultralytics import YOLO
# Auto-save logic here
"
EOF
```

### 4. Distributed Training (Multiple GPUs)

If you upgrade to multi-GPU VM (e.g., Standard_NC24):

```python
# In train.py, enable DDP
from ultralytics import YOLO

model = YOLO("./Model/yolo11m-seg.pt")

results = model.train(
    data="./dataset/YOLO/yolov11/data.yaml",
    epochs=100,
    device=[0, 1, 2, 3],  # Use 4 GPUs
    batch=96,  # 24 per GPU
    workers=16,
)
```

### 5. Model Checkpointing to Azure Blob Storage

```python
# Add to training script
import os
from azure.storage.blob import BlobServiceClient

def upload_checkpoint(model_path):
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service.get_container_client("model-checkpoints")
    
    with open(model_path, "rb") as data:
        blob_client = container.get_blob_client(os.path.basename(model_path))
        blob_client.upload_blob(data, overwrite=True)

# Use in callback
from ultralytics.utils.callbacks import Callbacks

def on_train_epoch_end(trainer):
    if trainer.epoch % 10 == 0:  # Every 10 epochs
        upload_checkpoint(trainer.best)

callbacks = Callbacks()
callbacks.add_callback("on_train_epoch_end", on_train_epoch_end)
```

---

## 📊 MONITORING AND LOGGING

### 1. Set Up Azure Monitor for VM

```bash
# Enable Azure Monitor
az vm extension set \
    --resource-group product-detection-rg \
    --vm-name product-detection-vm \
    --name AzureMonitorLinuxAgent \
    --publisher Microsoft.Azure.Monitor \
    --enable-auto-upgrade true
```

### 2. Streamlit Logging to File

```python
# Add to app.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/runs/streamlit.log'),
        logging.StreamHandler()
    ]
)
```

### 3. GPU Monitoring Dashboard

```bash
# Install Prometheus + Grafana (optional, advanced)
docker run -d --name prometheus \
    -p 9090:9090 \
    prom/prometheus

docker run -d --name grafana \
    -p 3000:3000 \
    grafana/grafana
```

---

## 🔐 PRODUCTION SECURITY HARDENING

### 1. Use Azure Key Vault for Secrets

```bash
# Create Key Vault
az keyvault create \
    --name productdetectionkv \
    --resource-group product-detection-rg \
    --location eastus

# Store ACR password
az keyvault secret set \
    --vault-name productdetectionkv \
    --name acr-password \
    --value "YOUR_ACR_PASSWORD"

# Retrieve in scripts
az keyvault secret show \
    --vault-name productdetectionkv \
    --name acr-password \
    --query value -o tsv
```

### 2. Enable Azure Firewall

```bash
# Create Network Security Group rules
az network nsg rule create \
    --resource-group product-detection-rg \
    --nsg-name product-detection-vmNSG \
    --name AllowSSHFromHome \
    --priority 100 \
    --source-address-prefixes YOUR_HOME_IP/32 \
    --destination-port-ranges 22 \
    --access Allow \
    --protocol Tcp

az network nsg rule create \
    --resource-group product-detection-rg \
    --nsg-name product-detection-vmNSG \
    --name AllowStreamlitFromHome \
    --priority 101 \
    --source-address-prefixes YOUR_HOME_IP/32 \
    --destination-port-ranges 8501 \
    --access Allow \
    --protocol Tcp
```

### 3. Enable Disk Encryption

```bash
# Create Key Vault for encryption keys
az keyvault create \
    --name diskencryptionkv \
    --resource-group product-detection-rg \
    --location eastus \
    --enabled-for-disk-encryption true

# Enable encryption
az vm encryption enable \
    --resource-group product-detection-rg \
    --name product-detection-vm \
    --disk-encryption-keyvault diskencryptionkv
```

---

## 🚀 DEPLOYMENT CHECKLIST

### Pre-Deployment:
- [ ] Test Docker image locally with GPU
- [ ] Verify CUDA and PyTorch versions match
- [ ] Test all Streamlit features work in container
- [ ] Prepare datasets and models
- [ ] Create `.dockerignore` file
- [ ] Document required environment variables

### Azure Setup:
- [ ] Create Azure Student account
- [ ] Install Azure CLI locally
- [ ] Login to Azure (`az login`)
- [ ] Check GPU VM quota
- [ ] Choose Azure region (eastus, westus2, etc.)

### Infrastructure Creation:
- [ ] Create Resource Group
- [ ] Create Azure Container Registry (ACR)
- [ ] Build and push Docker image to ACR
- [ ] Create GPU-enabled VM (NC4as_T4_v3)
- [ ] Open required ports (22, 8501)
- [ ] Set up auto-shutdown schedule

### VM Configuration:
- [ ] SSH into VM
- [ ] Update system packages
- [ ] Install Docker
- [ ] Install NVIDIA drivers
- [ ] Install NVIDIA Container Toolkit
- [ ] Verify GPU access (`nvidia-smi`)
- [ ] Pull Docker image from ACR
- [ ] Create persistent data directories

### Application Deployment:
- [ ] Run Docker container with GPU
- [ ] Verify container logs
- [ ] Test GPU inside container
- [ ] Upload datasets (SCP or Azure Storage)
- [ ] Test Streamlit UI access
- [ ] Run test training job

### Post-Deployment:
- [ ] Set up monitoring alerts
- [ ] Configure cost budgets
- [ ] Enable automatic backups
- [ ] Document public IP and access URLs
- [ ] Test model training end-to-end
- [ ] Set up SSL/HTTPS (optional)

### Ongoing Maintenance:
- [ ] Monitor Azure costs daily
- [ ] Deallocate VM when not in use
- [ ] Back up trained models regularly
- [ ] Update Docker images
- [ ] Review security logs

---

## 📚 ADDITIONAL RESOURCES

### Official Documentation:
- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **Azure GPU VMs**: https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu
- **Docker GPU Support**: https://docs.docker.com/config/containers/resource_constraints/#gpu
- **Streamlit**: https://docs.streamlit.io/
- **Azure Students**: https://azure.microsoft.com/en-us/free/students/

### Tutorials:
- PyTorch CUDA Setup: https://pytorch.org/get-started/locally/
- NVIDIA Docker: https://github.com/NVIDIA/nvidia-docker
- Azure CLI Reference: https://learn.microsoft.com/en-us/cli/azure/

### Community:
- Ultralytics Discord: https://ultralytics.com/discord
- Azure Forums: https://learn.microsoft.com/en-us/answers/
- Docker Community: https://forums.docker.com/

---

## 💡 FINAL TIPS

1. **Start Small**: Test on a small dataset first before full training
2. **Monitor Costs**: Check Azure portal daily for spending
3. **Use Spot Instances**: Save 70-90% on compute costs
4. **Automate Shutdowns**: Set up auto-shutdown when idle
5. **Version Control**: Keep track of model versions and configs
6. **Regular Backups**: Download trained models to local storage
7. **Security First**: Always restrict network access to your IP
8. **Documentation**: Keep notes on training parameters and results
9. **Experiment Tracking**: Use TensorBoard or Weights & Biases
10. **Ask for Help**: Azure student support is free and helpful

---

**Good luck with your product detection project! 🚀**

This documentation should serve as your complete reference for deploying GPU-enabled YOLO training on Azure. Keep it updated as you learn more!
