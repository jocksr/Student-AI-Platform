param([string]$OutputPath = 'results/hardware-windows.json')
$ErrorActionPreference = 'Stop'
$report = [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); source = 'read-only Windows CIM and command inventory' }
function Capture([scriptblock]$Action) {
    try { & $Action } catch { @{ error = $_.Exception.Message } }
}
$report.cpu = Capture { Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed }
$report.os = Capture { Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,TotalVisibleMemorySize,FreePhysicalMemory }
$report.ram = @(Capture { Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed,ConfiguredClockSpeed,Manufacturer,PartNumber,DeviceLocator,DataWidth })
$report.disks = @(Capture { Get-PhysicalDisk | Select-Object FriendlyName,BusType,MediaType,Size })
$report.volumes = @(Capture { Get-Volume | Select-Object DriveLetter,FileSystem,Size,SizeRemaining })
$report.partitions = @(Capture { Get-Partition | Select-Object DiskNumber,PartitionNumber,DriveLetter,Size })
$report.disk_numbers = @(Capture { Get-Disk | Select-Object Number,FriendlyName,BusType,Size })
$report.tools = @(Get-Command python,py,git,nvcc,nvidia-smi,cmake,cl,docker,wsl -ErrorAction SilentlyContinue | Select-Object Name,Source)
$report.wsl_status = Capture { (wsl --status 2>&1 | Out-String).Replace([string][char]0,'') }
$report.wsl_distributions = Capture { (wsl --list --verbose 2>&1 | Out-String).Replace([string][char]0,'') }
$report.docker_client = Capture { docker --version | Out-String }
$report.nvidia = Capture { nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv | Out-String }
$report.cuda_toolkit_folders = @(Get-ChildItem -LiteralPath 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -ErrorAction SilentlyContinue | Select-Object FullName)
$report.visual_studio_compilers = Capture {
    $vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) { & $vswhere -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe' } else { 'vswhere absent' }
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputPath) | Out-Null
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputPath -Encoding utf8
Write-Output $OutputPath
