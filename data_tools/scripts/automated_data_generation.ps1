# TS-SatFire 数据生成自动化脚本
# 这个脚本会自动化执行数据下载和预处理的大部分步骤

# ============================================================================
# 配置区域 - 请根据你的实际情况修改
# ============================================================================

$WORKSPACE_ROOT = "C:\Users\22125\Desktop\TS-SatFire-DataGen"
$KAGGLE_DATASET = "z789456sx/ts-satfire"
$CONDA_ENV_PREPROCESS = "ts-satfire-preprocess"
$FIRESET_PROJECT = "C:\Users\22125\Desktop\FireST-GraphNet"

# ============================================================================
# 步骤 1: 创建工作目录
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 1: Creating workspace directories" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

if (-not (Test-Path $WORKSPACE_ROOT)) {
    New-Item -ItemType Directory -Path $WORKSPACE_ROOT -Force | Out-Null
    Write-Host "[OK] Created workspace: $WORKSPACE_ROOT" -ForegroundColor Green
} else {
    Write-Host "[OK] Workspace already exists: $WORKSPACE_ROOT" -ForegroundColor Green
}

Set-Location $WORKSPACE_ROOT

# ============================================================================
# 步骤 2: 检查 Kaggle API 配置
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 2: Checking Kaggle API setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$kaggleJsonPath = "$env:USERPROFILE\.kaggle\kaggle.json"

if (Test-Path $kaggleJsonPath) {
    Write-Host "[OK] Kaggle API key found" -ForegroundColor Green
    
    # 测试 Kaggle API
    Write-Host "`nTesting Kaggle API connection..." -ForegroundColor Yellow
    try {
        kaggle datasets list --page-size 1 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] Kaggle API working" -ForegroundColor Green
        } else {
            throw "Kaggle API test failed"
        }
    } catch {
        Write-Host "[ERROR] Kaggle API not working properly" -ForegroundColor Red
        Write-Host "Please configure your Kaggle API key:" -ForegroundColor Yellow
        Write-Host "1. Visit https://www.kaggle.com/me/account" -ForegroundColor Yellow
        Write-Host "2. Click 'Create New API Token'" -ForegroundColor Yellow
        Write-Host "3. Save kaggle.json to: $kaggleJsonPath" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "[ERROR] Kaggle API key not found at: $kaggleJsonPath" -ForegroundColor Red
    Write-Host "`nPlease configure your Kaggle API:" -ForegroundColor Yellow
    Write-Host "1. Visit https://www.kaggle.com/me/account" -ForegroundColor Yellow
    Write-Host "2. Click 'Create New API Token'" -ForegroundColor Yellow
    Write-Host "3. Create directory: $env:USERPROFILE\.kaggle" -ForegroundColor Yellow
    Write-Host "4. Move kaggle.json to that directory" -ForegroundColor Yellow
    
    $response = Read-Host "`nWould you like to create the .kaggle directory now? (y/n)"
    if ($response -eq "y") {
        New-Item -ItemType Directory -Path "$env:USERPROFILE\.kaggle" -Force | Out-Null
        Write-Host "[OK] Created .kaggle directory" -ForegroundColor Green
        Write-Host "Please download kaggle.json and place it there, then run this script again." -ForegroundColor Yellow
    }
    exit 1
}

# ============================================================================
# 步骤 3: 下载原始数据（如果尚未下载）
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 3: Downloading raw GeoTIFF data" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$rawDataZip = Join-Path $WORKSPACE_ROOT "raw_data\ts-satfire.zip"
$rawDataDir = Join-Path $WORKSPACE_ROOT "raw_geotiff"

if (Test-Path $rawDataDir) {
    Write-Host "[OK] Raw data already downloaded: $rawDataDir" -ForegroundColor Green
    $response = Read-Host "Re-download? This will take 2-6 hours (y/n)"
    if ($response -ne "y") {
        Write-Host "Skipping download..." -ForegroundColor Yellow
    } else {
        Remove-Item -Recurse -Force $rawDataDir
    }
}

if (-not (Test-Path $rawDataDir)) {
    Write-Host "Downloading TS-SatFire dataset (~71GB)..." -ForegroundColor Yellow
    Write-Host "This will take 2-6 hours depending on your connection." -ForegroundColor Yellow
    Write-Host "You can leave this running in the background.`n" -ForegroundColor Yellow
    
    # 创建 raw_data 目录
    New-Item -ItemType Directory -Path (Join-Path $WORKSPACE_ROOT "raw_data") -Force | Out-Null
    
    # 下载数据集
    kaggle datasets download -d $KAGGLE_DATASET -p (Join-Path $WORKSPACE_ROOT "raw_data")
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n[OK] Download completed" -ForegroundColor Green
        
        # 解压
        Write-Host "`nExtracting archive..." -ForegroundColor Yellow
        Expand-Archive -Path $rawDataZip -DestinationPath $rawDataDir -Force
        Write-Host "[OK] Extraction completed" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Download failed" -ForegroundColor Red
        exit 1
    }
}

# ============================================================================
# 步骤 4: 克隆预处理代码
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 4: Cloning preprocessing code" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$preprocessCodeDir = Join-Path $WORKSPACE_ROOT "TS-SatFire"

if (Test-Path $preprocessCodeDir) {
    Write-Host "[OK] TS-SatFire repo already cloned" -ForegroundColor Green
} else {
    Write-Host "Cloning TS-SatFire repository..." -ForegroundColor Yellow
    git clone https://github.com/zhaoyutim/TS-SatFire.git $preprocessCodeDir
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Repository cloned" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Git clone failed" -ForegroundColor Red
        exit 1
    }
}

# ============================================================================
# 步骤 5: 创建预处理环境
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 5: Setting up preprocessing environment" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 检查环境是否已存在
$envExists = conda env list | Select-String -Pattern $CONDA_ENV_PREPROCESS

if ($envExists) {
    Write-Host "[OK] Conda environment '$CONDA_ENV_PREPROCESS' already exists" -ForegroundColor Green
    $response = Read-Host "Recreate environment? (y/n)"
    if ($response -eq "y") {
        Write-Host "Removing existing environment..." -ForegroundColor Yellow
        conda env remove -n $CONDA_ENV_PREPROCESS -y
        $envExists = $null
    }
}

if (-not $envExists) {
    Write-Host "Creating conda environment '$CONDA_ENV_PREPROCESS'..." -ForegroundColor Yellow
    conda create -n $CONDA_ENV_PREPROCESS python=3.9 -y
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Environment created" -ForegroundColor Green
        
        # 激活环境并安装依赖
        Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
        
        # 安装核心依赖
        conda run -n $CONDA_ENV_PREPROCESS pip install numpy pandas scikit-image opencv-python tqdm pyyaml
        conda run -n $CONDA_ENV_PREPROCESS pip install rasterio geopandas fiona shapely pyproj
        conda run -n $CONDA_ENV_PREPROCESS pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        
        Write-Host "[OK] Dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to create environment" -ForegroundColor Red
        exit 1
    }
}

# ============================================================================
# 步骤 6: 生成数据的交互式选择
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 6: Data generation configuration" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "The script can now generate the preprocessed data." -ForegroundColor Yellow
Write-Host "This will take 4-8 hours total (can run in background).`n" -ForegroundColor Yellow

Write-Host "What would you like to generate?" -ForegroundColor Cyan
Write-Host "1. Train + Val datasets (recommended)" -ForegroundColor White
Write-Host "2. Train + Val + Test datasets" -ForegroundColor White
Write-Host "3. Skip data generation (I'll do it manually)" -ForegroundColor White

$choice = Read-Host "`nEnter choice (1/2/3)"

if ($choice -eq "3") {
    Write-Host "`n[INFO] Skipping automatic data generation." -ForegroundColor Yellow
    Write-Host "To generate data manually, run:" -ForegroundColor Yellow
    Write-Host "  cd $preprocessCodeDir" -ForegroundColor Cyan
    Write-Host "  conda activate $CONDA_ENV_PREPROCESS" -ForegroundColor Cyan
    Write-Host "  python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc af" -ForegroundColor Cyan
    Write-Host "  python dataset_gen_afba.py -mode train -ts 1 -it 1 -uc ba" -ForegroundColor Cyan
    Write-Host "  python dataset_gen_pred.py -mode train -ts 1 -it 1" -ForegroundColor Cyan
    Write-Host "`nSee DATA_REGENERATION_STEP_BY_STEP.md for complete instructions.`n" -ForegroundColor Yellow
    exit 0
}

$modes = @("train", "val")
if ($choice -eq "2") {
    $modes += "test"
}

Write-Host "`n[INFO] Will generate: $($modes -join ', ') datasets" -ForegroundColor Green
Write-Host "[INFO] Estimated time: $(if ($choice -eq '1') {'4-6'} else {'6-10'}) hours" -ForegroundColor Yellow
Write-Host "[INFO] You can leave this running in the background.`n" -ForegroundColor Yellow

$response = Read-Host "Continue? (y/n)"
if ($response -ne "y") {
    Write-Host "Exiting..." -ForegroundColor Yellow
    exit 0
}

# ============================================================================
# 步骤 7: 运行数据生成命令
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 7: Generating preprocessed datasets" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Set-Location $preprocessCodeDir

foreach ($mode in $modes) {
    Write-Host "`n>>> Generating $mode dataset (AF + BA + Pred)..." -ForegroundColor Cyan
    Write-Host "    This will take 1-3 hours...`n" -ForegroundColor Yellow
    
    # AF
    Write-Host "[1/3] Generating AF $mode data..." -ForegroundColor Yellow
    conda run -n $CONDA_ENV_PREPROCESS python dataset_gen_afba.py -mode $mode -ts 1 -it 1 -uc af
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] AF $mode generation failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] AF $mode completed`n" -ForegroundColor Green
    
    # BA
    Write-Host "[2/3] Generating BA $mode data..." -ForegroundColor Yellow
    conda run -n $CONDA_ENV_PREPROCESS python dataset_gen_afba.py -mode $mode -ts 1 -it 1 -uc ba
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] BA $mode generation failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] BA $mode completed`n" -ForegroundColor Green
    
    # Pred
    Write-Host "[3/3] Generating Pred $mode data..." -ForegroundColor Yellow
    conda run -n $CONDA_ENV_PREPROCESS python dataset_gen_pred.py -mode $mode -ts 1 -it 1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Pred $mode generation failed" -ForegroundColor Red
        exit 1
    }
    Write-Host "[OK] Pred $mode completed`n" -ForegroundColor Green
    
    Write-Host "[SUCCESS] $mode dataset generation completed!`n" -ForegroundColor Green
}

# ============================================================================
# 步骤 8: 验证生成的数据
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 8: Validating generated data" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 找到输出目录（可能需要根据实际脚本调整）
$outputDir = Join-Path $preprocessCodeDir "output"

if (Test-Path $outputDir) {
    Write-Host "Running validation on generated data...`n" -ForegroundColor Yellow
    
    # 复制验证脚本
    $verifyScript = Join-Path $FIRESET_PROJECT "data_tools\validation\quick_verify.py"
    if (Test-Path $verifyScript) {
        Copy-Item $verifyScript $outputDir -Force
        
        Set-Location $outputDir
        conda run -n $CONDA_ENV_PREPROCESS python quick_verify.py
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`n[SUCCESS] Data validation passed!" -ForegroundColor Green
        } else {
            Write-Host "`n[WARNING] Data validation found issues" -ForegroundColor Yellow
            Write-Host "Please check the output above for details." -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "[WARNING] Output directory not found: $outputDir" -ForegroundColor Yellow
    Write-Host "Please verify the output location in the preprocessing scripts." -ForegroundColor Yellow
}

# ============================================================================
# 步骤 9: 复制数据到 FireST-GraphNet 项目
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Step 9: Copying data to project directory" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$response = Read-Host "Copy generated data to FireST-GraphNet project? (y/n)"

if ($response -eq "y") {
    # 备份旧数据
    $backupDir = Join-Path $FIRESET_PROJECT "old_data_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
    
    Write-Host "Backing up old data to: $backupDir" -ForegroundColor Yellow
    Copy-Item (Join-Path $FIRESET_PROJECT "dataset_train\*.npy") $backupDir -Force -ErrorAction SilentlyContinue
    Copy-Item (Join-Path $FIRESET_PROJECT "dataset_val\*.npy") $backupDir -Force -ErrorAction SilentlyContinue
    
    # 复制新数据
    Write-Host "Copying new data..." -ForegroundColor Yellow
    
    foreach ($mode in $modes) {
        $destDir = Join-Path $FIRESET_PROJECT "dataset_$mode"
        
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        
        # 列出需要复制的文件
        $filesToCopy = @(
            "af_${mode}_img_*.npy",
            "af_${mode}_label_*.npy",
            "firepred_${mode}.npy",
            "lulc_${mode}.npy"
        )
        
        foreach ($pattern in $filesToCopy) {
            $files = Get-ChildItem -Path $outputDir -Filter $pattern
            foreach ($file in $files) {
                Copy-Item $file.FullName $destDir -Force
                Write-Host "  [OK] Copied $($file.Name)" -ForegroundColor Green
            }
        }
    }
    
    Write-Host "`n[SUCCESS] Data copied to FireST-GraphNet project!" -ForegroundColor Green
}

# ============================================================================
# 完成
# ============================================================================

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "DATA GENERATION COMPLETED!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "- Raw data location: $rawDataDir" -ForegroundColor White
Write-Host "- Preprocessing code: $preprocessCodeDir" -ForegroundColor White
Write-Host "- Generated data: $outputDir" -ForegroundColor White
Write-Host "- Project datasets: $FIRESET_PROJECT\dataset_*" -ForegroundColor White

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Run final verification in FireST-GraphNet:" -ForegroundColor Yellow
Write-Host "   cd $FIRESET_PROJECT" -ForegroundColor White
Write-Host "   conda activate wildfire_gnn" -ForegroundColor White
Write-Host "   python data_tools\validation\quick_verify.py" -ForegroundColor White
Write-Host "`n2. Upload data to Google Drive for Colab training" -ForegroundColor Yellow
Write-Host "`n3. Start multi-task training!" -ForegroundColor Yellow

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
