#!/usr/bin/env bash
# DEBIAN/UBUNTU INSTALLATION SCRIPT

# ensure APT is updated
sudo apt update

# setup the Github CLI repository
(type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \

# setup the Docker repository
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

# update APT
sudo apt update

# install Github CLI
sudo apt install gh -y

# install Docker
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

# install Nektos ACT
if ! gh auth status >/dev/null 2>&1; then
	# if interactive, login
	if [[ -t 0 && -t 1 ]]; then
		gh auth login
    # otherwise give a better error message
	else
        echo "error: GitHub CLI ('gh') is installed but not authenticated." >&2
        echo >&2
        echo "This script needs an authenticated gh session to install the act extension." >&2
        echo "Fix: run 'gh auth login' (or set GH_TOKEN for non-interactive use), then re-run:" >&2
        echo "  gh extension install https://github.com/nektos/gh-act" >&2
        echo >&2
		exit 1
	fi
fi
gh extension install https://github.com/nektos/gh-act
