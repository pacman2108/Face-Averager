param(
    [Parameter(Mandatory=$true)]
    [string]$InputPath
)

# Create and ensure necessary directories
$dirs = @("ims", "faces", "frames")
foreach ($d in $dirs) {
    if (-not (Test-Path -Path $d)) {
        New-Item -ItemType Directory -Path $d | Out-Null
    }
}

# Fixed URLs
$MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
$AVGPY_URL = 'https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceAverage/faceAverage.py'

# Ensure cmake is available
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "[!] cmake is needed. Install via your package manager, e.g. 'sudo apt install cmake'"
    Write-Host "[!] Once installed, rerun this script."
    exit 1
}

# Ensure Python packages dlib and scikit-image are installed
$packages = (& pip3 list --format=columns).Split([Environment]::NewLine) |
            Where-Object { $_ -match '^(dlib|scikit-image)\s' }
if ($packages.Count -ne 2) {
    Write-Host "[!] Installing missing Python packages dlib and scikit-image..."
    & pip3 install -q dlib scikit-image
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Failed to install packages. Run: pip3 install dlib scikit-image"
        exit 1
    }
}

# Download face landmark model if missing
if (-not (Test-Path -Path "model.dat")) {
    Write-Host "[-] Grabbing the NN face detection model (~100 MB)"
    Invoke-WebRequest -Uri $MODEL_URL -OutFile "model.dat.bz2" -UseBasicParsing
    Write-Host "[-] Decompressing model.dat.bz2"
    & bzip2 -d model.dat.bz2
} else {
    Write-Host "[+] Model already downloaded"
}

# Download and patch faceAverage.py if missing
if (-not (Test-Path -Path "faceAverage.py")) {
    Write-Host "[-] Downloading faceAverage.py"
    Invoke-WebRequest -Uri $AVGPY_URL -OutFile "faceAverage.py" -UseBasicParsing
    Write-Host "[-] Applying custom patches"
    (Get-Content faceAverage.py) |
        ForEach-Object {
            $_ -replace '600', '1000' `
               -replace 'cv2.waitKey\(0\)', '' `
               -replace 'presidents', 'faces' `
               -replace "cv2.imshow\('image', output\)", "cv2.imwrite('average.png', 255*output)" `
               -replace 'img = np.z', 'Write-Host "[+] Aligning faces..."; img = np.z' `
               -replace 'filePath.endsWith\(".jpg"\)', 'filePath.endswith(".jpg") -or filePath.endswith(".jpeg") -or filePath.endswith(".png")'
        } | Set-Content faceAverage.py
} else {
    Write-Host "[+] Already have faceAverage.py"
}

Write-Host "[+] Environment setup complete`n"

# Process input
if (Test-Path -Path $InputPath -PathType Leaf) {
    Write-Host "[+] Averaging faces from video: $InputPath`n"
    Write-Host "[-] Extracting every 5th frame..."
    ffmpeg -nostats -loglevel 0 -i $InputPath -vf "select=not(mod(n\,5))" -vsync vfr -q:v 2 "frames/img_%05d.jpg"
    python3 facePlucker.py
} elseif (Test-Path -Path $InputPath -PathType Container) {
    Write-Host "[+] Averaging faces from pictures in: $InputPath`n"
    Copy-Item -Path "$InputPath\*" -Destination ims -Recurse
} else {
    Write-Host "[!] Provide a video file or directory of images containing faces"
    exit 1
}

# Shape faces
python3 faceShaper.py

# Copy processed images into faces
Get-ChildItem -Path ims -Include *.png,*.jpg,*.jpeg,*.txt -Recurse |
    Copy-Item -Destination faces -Force

# Average faces
python3 faceAverage.py

# Clean up
Remove-Item -Recurse -Force ims, faces, frames

Write-Host "average.png written, enjoy!"
