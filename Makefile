.DEFAULT_GOAL := help

COMPOSE ?= docker compose
DEV_COMPOSE_FILE ?= docker-compose.dev.yaml
PROD_COMPOSE_FILE ?= docker-compose.yaml
DEV_PROFILE ?= webui

DEV_COMPOSE := $(COMPOSE) -f $(DEV_COMPOSE_FILE) --profile $(DEV_PROFILE)
PROD_COMPOSE := $(COMPOSE) -f $(PROD_COMPOSE_FILE)

.PHONY: help dev dev-bg dev-down dev-logs dev-ps dev-restart-api dev-restart-webui dev-shell-api dev-shell-webui
.PHONY: up up-webui down logs ps

help: ## Show available commands.
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*## "}; {printf "%-22s %s\n", $$1, $$2}'

dev: ## Start API + WebUI in dev mode with hot reloading (foreground).
	$(DEV_COMPOSE) up --build --remove-orphans

dev-bg: ## Start API + WebUI in dev mode with hot reloading (background).
	$(DEV_COMPOSE) up --build -d --remove-orphans

dev-down: ## Stop and remove dev containers, networks, and volumes.
	$(DEV_COMPOSE) down -v

dev-logs: ## Tail dev logs for API and WebUI.
	$(DEV_COMPOSE) logs -f --tail=200 api webui

dev-ps: ## Show dev service status.
	$(DEV_COMPOSE) ps

dev-restart-api: ## Restart dev API service.
	$(DEV_COMPOSE) restart api

dev-restart-webui: ## Restart dev WebUI service.
	$(DEV_COMPOSE) restart webui

dev-shell-api: ## Open a shell in the dev API container.
	$(DEV_COMPOSE) exec api bash

dev-shell-webui: ## Open a shell in the dev WebUI container.
	$(DEV_COMPOSE) exec webui bash

up: ## Start production-style API service in background.
	$(PROD_COMPOSE) up --build -d api

up-webui: ## Start production-style API + WebUI in background.
	$(PROD_COMPOSE) --profile webui up --build -d

down: ## Stop and remove production-style containers, networks, and volumes.
	$(PROD_COMPOSE) down -v

logs: ## Tail production-style logs.
	$(PROD_COMPOSE) logs -f --tail=200

ps: ## Show production-style service status.
	$(PROD_COMPOSE) ps
