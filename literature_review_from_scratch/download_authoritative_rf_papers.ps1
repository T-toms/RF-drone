$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$targetDir = 'C:\Users\Administrator\Desktop\数据集处理\literature_review_from_scratch\downloaded_papers'
$existingDir = 'C:\Users\Administrator\Desktop\文献'

$papers = @(
    [pscustomobject]@{ N = '01'; Doi = '10.1109/TIFS.2024.3463535'; File = '01_TIFS2024_Open_Set_Learning_RF_Drone_Recognition.pdf'; Fallback = @('https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10684814') },
    [pscustomobject]@{ N = '02'; Doi = '10.1109/JSAC.2026.3669136'; File = '02_JSAC2026_Multi_Domain_Contrastive_RF_Open_Set.pdf'; Fallback = @('https://arxiv.org/pdf/2508.12689') },
    [pscustomobject]@{ N = '03'; Doi = '10.1109/TIFS.2026.3677579'; File = '03_TIFS2026_Interpretability_Frequency_Aware_UAV_Recognition.pdf'; Fallback = @() },
    [pscustomobject]@{ N = '04'; Doi = '10.1145/3372224.3419189'; File = '04_MobiCom2020_SDR_Lite_Physical_Layer_Reconstruction.pdf'; Fallback = @('https://seit.egr.msu.edu/paper/Mobicom2020-SDRLite.pdf') },
    [pscustomobject]@{ N = '05'; Doi = '10.1109/TVT.2020.3042128'; File = '05_TVT2020_RF_Fingerprinting_Nonstandard_UAV_Waveforms.pdf'; Fallback = @('https://genesys-lab.org/papers/UAV-TVT-20.pdf', 'https://nasimsoltani.github.io/files/uav.pdf') },
    [pscustomobject]@{ N = '06'; Doi = '10.1109/TCCN.2021.3099114'; File = '06_TCCN2022_Combined_RF_Drone_Detection_Classification.pdf'; Fallback = @('https://d197for5662m48.cloudfront.net/documents/publicationstatus/161337/preprint_pdf/169c345bd18473a63f3f0f03b677bcec.pdf') },
    [pscustomobject]@{ N = '07'; Doi = '10.1016/j.future.2019.05.007'; File = '07_FGCS2019_RF_Drone_Detection_Identification_DL.pdf'; Fallback = @() },
    [pscustomobject]@{ N = '08'; Doi = '10.1109/TIM.2023.3306822'; File = '08_TIM2023_RF_Identification_Nonstandard_Waveforms.pdf'; Fallback = @() },
    [pscustomobject]@{ N = '09'; Doi = '10.1109/TR.2021.3119068'; File = '09_TR2022_Reliable_UAV_Monitoring_DL.pdf'; Fallback = @() },
    [pscustomobject]@{ N = '10'; Doi = '10.1016/j.eswa.2021.115928'; File = '10_ESWA2022_Single_Multiple_Drones_RF_DL.pdf'; Fallback = @() },
    [pscustomobject]@{ N = '11'; Doi = '10.1016/j.comnet.2021.108519'; File = '11_ComputerNetworks2021_CNN_SSDI.pdf'; Fallback = @() },
    [pscustomobject]@{ N = '12'; Doi = '10.1109/TVT.2021.3081049'; File = '12_TVT2021_Proprietary_Waveforms_Preamble_Federated_Learning.pdf'; Fallback = @('https://genesys-lab.org/papers/Classifying_UAVs_TVT.pdf') }
)

function Test-PdfFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -lt 10000) { return $false }
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $bytes = New-Object byte[] 5
        $read = $stream.Read($bytes, 0, 5)
        if ($read -ne 5) { return $false }
        return ([System.Text.Encoding]::ASCII.GetString($bytes) -eq '%PDF-')
    }
    finally {
        $stream.Dispose()
    }
}

function Get-OpenAlexPdfUrls {
    param([Parameter(Mandatory = $true)][string]$Doi)
    $workUrl = "https://api.openalex.org/works/https://doi.org/$Doi"
    try {
        $work = Invoke-RestMethod -Uri $workUrl -UseBasicParsing -Headers @{ 'User-Agent' = 'Codex literature downloader/1.0' } -TimeoutSec 10
    }
    catch {
        return @()
    }
    $urls = New-Object System.Collections.Generic.List[string]
    if ($null -ne $work.best_oa_location -and -not [string]::IsNullOrWhiteSpace([string]$work.best_oa_location.pdf_url)) {
        $urls.Add([string]$work.best_oa_location.pdf_url)
    }
    foreach ($location in @($work.locations)) {
        if ($null -ne $location -and -not [string]::IsNullOrWhiteSpace([string]$location.pdf_url)) {
            $urls.Add([string]$location.pdf_url)
        }
    }
    return @($urls | Select-Object -Unique)
}

New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
$results = New-Object System.Collections.Generic.List[object]

foreach ($paper in $papers) {
    $destination = Join-Path $targetDir $paper.File
    if (Test-PdfFile -Path $destination) {
        $results.Add([pscustomobject]@{ No = $paper.N; DOI = $paper.Doi; Status = 'already-present'; File = $paper.File; Source = '' })
        continue
    }

    if ($paper.Doi -eq '10.1109/TCCN.2021.3099114') {
        $existing = Join-Path $existingDir '04_Combined_RF-Based_Drone_Detection_and_Classification_TCCN2022_preprint.pdf'
        if (Test-PdfFile -Path $existing) {
            $results.Add([pscustomobject]@{ No = $paper.N; DOI = $paper.Doi; Status = 'existing-equivalent'; File = (Split-Path -Leaf $existing); Source = $existing })
            continue
        }
    }

    $candidateUrls = @((Get-OpenAlexPdfUrls -Doi $paper.Doi)) + @($paper.Fallback)
    $downloaded = $false
    $usedUrl = ''
    foreach ($url in @($candidateUrls | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
        $temporary = "$destination.part"
        try {
            Invoke-WebRequest -Uri $url -OutFile $temporary -UseBasicParsing -MaximumRedirection 8 -Headers @{ 'User-Agent' = 'Mozilla/5.0' } -TimeoutSec 45
            if (Test-PdfFile -Path $temporary) {
                Move-Item -LiteralPath $temporary -Destination $destination -Force
                $downloaded = $true
                $usedUrl = $url
                break
            }
        }
        catch {
            # Try the next legal open-access location.
        }
        finally {
            if (Test-Path -LiteralPath $temporary) {
                Remove-Item -LiteralPath $temporary -Force
            }
        }
    }

    $results.Add([pscustomobject]@{
        No = $paper.N
        DOI = $paper.Doi
        Status = if ($downloaded) { 'downloaded' } else { 'no-open-pdf-found' }
        File = if ($downloaded) { $paper.File } else { '' }
        Source = $usedUrl
    })
}

$results | Sort-Object No | ConvertTo-Json -Depth 3
