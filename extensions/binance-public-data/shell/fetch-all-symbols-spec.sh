#!/bin/bash

# Simple script to download all symbols specifications from https://api.binance.com/api/v3/exchangeInfo

curl -s -H 'Content-Type: application/json'  https://api.binance.com/api/v3/exchangeInfo > symbol.json
