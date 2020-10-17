#!/bin/bash

ps -ef | grep "runspec" | awk '{print $2;}' | xargs -r kill
