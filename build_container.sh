#!/bin/bash
docker build -t roskamsh/nopeak-utils:latest --secret id=gitlab,src=credentials.sh .