$neo = Test-NetConnection -ComputerName localhost -Port 7687 -WarningAction SilentlyContinue
$qdr = Test-NetConnection -ComputerName localhost -Port 6333 -WarningAction SilentlyContinue

Write-Host "Neo4j 7687: $($neo.TcpTestSucceeded)"
Write-Host "Qdrant 6333: $($qdr.TcpTestSucceeded)"

if (-not $neo.TcpTestSucceeded -or -not $qdr.TcpTestSucceeded) {
    Write-Host "Au moins un service local n'est pas joignable." -ForegroundColor Yellow
    exit 1
}

Write-Host "Stack local joignable." -ForegroundColor Green
