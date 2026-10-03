# Makefile — one target per stage; reproduce.sh drives these (and can also be
# called directly via `./reproduce.sh --stage <name>`).

.PHONY: all setup data weights manifest train eval export app bench dry-run

all:
	./reproduce.sh

setup:
	./reproduce.sh --stage setup

data:
	./reproduce.sh --stage data

weights:
	./reproduce.sh --stage weights

manifest:
	./reproduce.sh --stage manifest

train:
	./reproduce.sh --train dgx

eval:
	./reproduce.sh --stage eval

export:
	./reproduce.sh --stage export

app:
	./reproduce.sh --stage app

bench:
	./reproduce.sh --pi $(PI_HOST)

dry-run:
	./reproduce.sh --dry-run
