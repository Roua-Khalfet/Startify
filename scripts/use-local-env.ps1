$env:NEO4J_URI = "bolt://localhost:7687"
$env:NEO4J_USERNAME = "neo4j"
$env:NEO4J_PASSWORD = "neo4j123"
$env:NEO4J_DATABASE = "neo4j"

$env:QDRANT_URL = "http://localhost:6333"
$env:QDRANT_API_KEY = ""
$env:QDRANT_PATH = ""
$env:QDRANT_COLLECTION_NAME = "complianceguard_chunks"
$env:QDRANT_USER_COLLECTION_NAME = "user_uploads"

Write-Host "Local env active pour cette session PowerShell." -ForegroundColor Green
Write-Host "NEO4J_URI=$env:NEO4J_URI"
Write-Host "QDRANT_URL=$env:QDRANT_URL"
