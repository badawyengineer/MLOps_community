install:
	python -m pip install -e ".[dev]"

run:
	uvicorn legal_rag.api:app --reload

test:
	pytest

coverage:
	pytest --cov=src/legal_rag --cov-report=html --cov-fail-under=80

docker-up:
	docker compose up --build

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api
