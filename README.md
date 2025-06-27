# german-legal-reference-ner


## Install / Compile
To run, first create a virtual environment using `python3 -m venv --prompt legal-citation-ner .venv`. You can then activate it using `source .venv/bin/activate`. Next update pip/wheel using `pip install --upgrade pip wheel`. After that install all required packages `pip install -r requirements.txt`.

## Labelling
To label a sentence you need to run either

    python label.py classify "einschließlich der anzurechnenden Untersuchungshaft, vgl. BGH NStZ-RR 2008, 182"

if you want to tag the is-citation-or-not or

    python label.py "einschließlich der anzurechnenden Untersuchungshaft, vgl. BGH NStZ-RR 2008, 182"

if you want the full labels
