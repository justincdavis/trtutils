#!/usr/bin/env bash

find . -name '*.py' -exec pyupgrade {} +
