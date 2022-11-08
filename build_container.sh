#!/bin/bash
docker build -t oalmelid/nopeak-utils:latest --secret id=gitlab,src=credentials.sh .