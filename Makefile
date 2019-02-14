.PHONY: autodoc doc docopen docinit docopen init install test clean

autodoc:
	rm -rf docs/source
	sphinx-apidoc -eMT -o docs/source/ gpar
	rm docs/source/gpar.rst
	pandoc --from=markdown --to=rst --output=docs/readme.rst README.md

doc:
	cd docs && make html

docopen:
	open docs/_build/html/index.html

docinit:
	git checkout -b gh-pages
	git ls-tree HEAD \
		| awk '$4 !~ /\.gitignore|\.nojekyll|Makefile|docs|index\.html/ { print $4 }' \
		| xargs -I {} git rm -r {}
	touch .nojekyll
	git add .nojekyll
	echo '<meta http-equiv="refresh" content="0; url=./docs/_build/html/index.html" />\n' > index.html
	git commit -m "Branch cleaned for docs"

docupdate: autodoc doc
	git checkout gh-pages
	git add -f docs/_build/html
	git commit -m "Update docs at $(date +'%d %b %Y, %H:%M')"
	git push origin gh-pages

init:
	pip install -r requirements.txt

install:
	pip install -r requirements.txt -e .

test:
	python /usr/local/bin/nosetests tests --with-coverage --cover-html --cover-package=gpar -v --logging-filter=gpar

clean:
	rm -rf docs/_build docs/source docs/readme.rst
	git rm --cached -r docs
	git add docs/Makefile docs/conf.py docs/index.rst docs/api.rst
	rm -rf .coverage cover
	find . | grep '\(\.DS_Store\|\.pyc\)$$' | xargs rm
