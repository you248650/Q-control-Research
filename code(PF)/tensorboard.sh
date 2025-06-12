#!/bin/bash

logdir=$1

tensorboard --logdir logdir --port 6006 --bind_all &
firefox http://localhost:6006/