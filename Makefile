.PHONY : clean
clean :
	rm -f figures/*.png
	rm -f data/*Error*


.PHONY : env
env:
	mamba env create -f environment.yml --name housetools
	bash -ic "conda activate housetools"
	pip install scikit-learn
	pip install .
	bash -ic 'python -m ipykernel install --user --name housetools --display-name "IPython -housetools"'

.PHONY : html
html:
	jupyter-book build .
    
.PHONY : html-hub
html-hub:
	sphinx-build  . _build/html -D html_baseurl=${JUPYTERHUB_SERVICE_PREFIX}/proxy/absolute/8000
	@echo "Start the Python http server and visit:
	@echo "https://stat159.datahub.berkeley.edu/user-redirect/proxy/8000/index.html"


.PHONY : all
all:
	jupyter execute combined.ipynb
	jupyter execute main.ipynb

