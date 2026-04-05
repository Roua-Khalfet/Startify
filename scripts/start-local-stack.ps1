param(
    [string]$Neo4jPassword = "neo4j123",
    [string]$DockerExe = "",
    [switch]$RecreateNeo4j
)

$ErrorActionPreference = "Stop"

function Resolve-DockerCli {
    param([string]$PreferredPath = "")

    if ($PreferredPath -and (Test-Path $PreferredPath)) {
        return (Resolve-Path $PreferredPath).Path
    }

    $cmd = Get-Command docker -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "$Env:ProgramFiles\Docker\Docker\resources\bin\docker.exe",
        "$Env:ProgramW6432\Docker\Docker\resources\bin\docker.exe",
        "$Env:LocalAppData\Docker\Docker\resources\bin\docker.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

$DockerCli = Resolve-DockerCli -PreferredPath $DockerExe
if (-not $DockerCli) {
    Write-Host "Docker CLI introuvable dans cette session VS Code." -ForegroundColor Yellow
    Write-Host "Actions suggerees:" -ForegroundColor Yellow
    Write-Host "  1) Ferme et rouvre VS Code (ou Developer: Reload Window)" -ForegroundColor Yellow
    Write-Host "  2) Ou lance avec -DockerExe \"C:\Program Files\Docker\Docker\resources\bin\docker.exe\"" -ForegroundColor Yellow
    exit 1
}

Write-Host "Docker CLI detecte: $DockerCli" -ForegroundColor DarkGray

# Le Docker CLI peut invoquer des helpers (ex: docker-credential-desktop)
# qui doivent etre resolvables via PATH.
$dockerBin = Split-Path -Parent $DockerCli
if ($env:PATH -notlike "*$dockerBin*") {
    $env:PATH = "$dockerBin;$env:PATH"
}

function Ensure-Container {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$Image,
        [Parameter(Mandatory=$true)][string[]]$RunArgs
    )

    $exists = & $DockerCli ps -a --filter "name=^/${Name}$" --format "{{.Names}}"
    if (-not $exists) {
        Write-Host "Creation du conteneur $Name..."
        $runArgs = @("run", "-d", "--name", $Name) + $RunArgs + @($Image)
        & $DockerCli @runArgs | Out-Null
    } else {
        Write-Host "Demarrage du conteneur $Name..."
        & $DockerCli start $Name | Out-Null
    }
}

function Remove-ContainerIfRequested {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][bool]$ShouldRemove
    )

    if (-not $ShouldRemove) {
        return
    }

    $exists = & $DockerCli ps -a --filter "name=^/${Name}$" --format "{{.Names}}"
    if ($exists) {
        Write-Host "Suppression du conteneur $Name (recreate demande)..."
        & $DockerCli rm -f $Name | Out-Null
    }
}

Remove-ContainerIfRequested -Name "neo4j-local" -ShouldRemove ([bool]$RecreateNeo4j)

Ensure-Container -Name "neo4j-local" -Image "neo4j:5" -RunArgs @(
    "-p", "7474:7474",
    "-p", "7687:7687",
    "-e", "NEO4J_AUTH=neo4j/$Neo4jPassword",
    "-e", 'NEO4J_PLUGINS=["apoc"]',
    "-e", 'NEO4J_dbms_security_procedures_unrestricted=apoc.*',
    "-e", 'NEO4J_dbms_security_procedures_allowlist=apoc.*'
)

Ensure-Container -Name "qdrant-local" -Image "qdrant/qdrant" -RunArgs @(
    "-p", "6333:6333",
    "-p", "6334:6334"
)

Write-Host ""
Write-Host "Etat des conteneurs:" -ForegroundColor Cyan
& $DockerCli ps --filter "name=neo4j-local" --filter "name=qdrant-local" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

Write-Host ""
Write-Host "URLs:" -ForegroundColor Cyan
Write-Host "Neo4j Browser: http://localhost:7474"
Write-Host "Qdrant API:    http://localhost:6333/collections"

if ($RecreateNeo4j) {
    Write-Host ""
    Write-Host "Neo4j a ete recree avec APOC active." -ForegroundColor Yellow
}
