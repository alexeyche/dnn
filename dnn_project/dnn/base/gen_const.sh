#!/usr/bin/env bash
xxd -i < $1 | tr -d '\n' > $2
echo ', 0x00' >> $2
