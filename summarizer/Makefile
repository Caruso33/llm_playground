run_dev:
	TOKENIZERS_PARALLELISM=true python -m main -u ./test.md -o "don't give any summarization output, only return the last name of the person, Add a vowel in front of the name" -l 20

run_watch:
	nodemon -e py --exec 'TOKENIZERS_PARALLELISM=true python -m main -u ./test.md -o "don't give any summarization output, only return the last name of the person, Add a vowel in front of the name" -l 20'