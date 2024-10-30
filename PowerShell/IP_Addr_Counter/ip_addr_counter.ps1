# Define the script
function Get-TopIPAddresses {
  param (
    [int]$n,
    [string]$d,
    [string]$m,
    [string]$y
  )

  # Read the log file
  $logEntries = Get-Content -Path "logs.txt"

  # Filter log entries by date if a date range is provided
  if ($d) {
    $date = Get-Date -Date $d
    $logEntries = $logEntries | Where-Object { $_ -match "\[(\d{2}\/\w{3}\/\d{4})" } | Where-Object { (Get-Date -Date $Matches[1]) -eq $date }
  } elseif ($m) {
    $logEntries = $logEntries | Where-Object { $_ -match "\[(\d{2}\/\w{3}\/\d{4})" } | Where-Object { (Get-Date -Date $Matches[1]).Month -eq [int]$m }
  } elseif ($y) {
    $logEntries = $logEntries | Where-Object { $_ -match "\[(\d{2}\/\w{3}\/\d{4})" } | Where-Object { (Get-Date -Date $Matches[1]).Year -eq [int]$y }
  }

  # Extract IP addresses from log entries
  $ipAddresses = $logEntries | ForEach-Object { $_.Split()[0] }

  # Count occurrences of each IP address
  $ipAddressCounts = $ipAddresses | Group-Object | Sort-Object -Property Count -Descending

  # Print top n IP addresses
  $ipAddressCounts | Select-Object -First $n | ForEach-Object { Write-Host "$($_.Name): $($_.Count)" }
}

# Call the script with the desired arguments
Get-TopIPAddresses -n 10 -m "11"