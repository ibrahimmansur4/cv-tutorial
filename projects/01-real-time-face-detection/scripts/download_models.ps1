[CmdletBinding()]
param(
    [Parameter()]
    [string]$Destination = (Join-Path (Split-Path -Parent $PSScriptRoot) "models"),

    [Parameter()]
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# These URLs are maintained by the OpenCV project. The 4.x text files remain
# compatible with the Caffe model stored on OpenCV's dedicated third-party-data
# branch.
$downloads = @(
    [pscustomobject]@{
        Name = "haarcascade_frontalface_default.xml"
        Uri = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml"
        MinimumBytes = 100000
    },
    [pscustomobject]@{
        Name = "deploy.prototxt"
        Uri = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
        MinimumBytes = 10000
    },
    [pscustomobject]@{
        Name = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
        Uri = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        MinimumBytes = 5000000
    }
)

$destinationPath = [System.IO.Path]::GetFullPath($Destination)
[System.IO.Directory]::CreateDirectory($destinationPath) | Out-Null

foreach ($download in $downloads) {
    $targetPath = Join-Path $destinationPath $download.Name
    $temporaryPath = "$targetPath.download"

    if ((Test-Path -LiteralPath $targetPath -PathType Leaf) -and -not $Force) {
        $existingLength = (Get-Item -LiteralPath $targetPath).Length
        if ($existingLength -ge $download.MinimumBytes) {
            Write-Host "Already present: $targetPath"
            continue
        }

        throw "Existing file appears incomplete: $targetPath. Re-run with -Force to replace it."
    }

    try {
        Write-Host "Downloading $($download.Name)"
        Invoke-WebRequest -Uri $download.Uri -OutFile $temporaryPath -UseBasicParsing

        $downloadedLength = (Get-Item -LiteralPath $temporaryPath).Length
        if ($downloadedLength -lt $download.MinimumBytes) {
            throw "Download for $($download.Name) is unexpectedly small ($downloadedLength bytes)."
        }

        Move-Item -LiteralPath $temporaryPath -Destination $targetPath -Force
        Write-Host "Saved: $targetPath ($downloadedLength bytes)"
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}

Write-Host "Model setup complete."
