# EuroMov DHM LLOD Knowledge Extractor
This is a first prototype that will be the basis of further development, but it is fully functional in terms of what is described in the upcoming preprint.
You may find the full evaluation results for the described use-case at the end of this README. Likewise, wou may find the final contents of the output here: [paper_output_reco315_rapport_lombalgie_2019_04_02_extraction.zip](https://github.com/EuromovDHM-SemTaxM/EuroMov-PFE-Lombalgie-KG/raw/master/paper_output_reco315_rapport_lombalgie_2019_04_02_extraction.zip). 

## I. Dependencies
This software requires Python 3.9+ to run. Please first install the dependences in `requirements.txt` by running `pip install -r requirements.txt` from the cloned project directory.

## II. Main scripts of the framework
### II.1. PDF Extraction 
PDF Extracction is performed by the `1_extract_text_from_pdfs.py`, which can be executed as `python 1_extract_text_from_pdfs.py input_path --target target_path`, 
where: 
 - `input_path` is either a single pdf file or a directory containing pdf files 
 - `--target target_path` is an optional parameter indicating the target directory where the result of the extraction is saved. By default, this is `./extracted_data`. If the input was a file, a single sub directory is created, holding for name the base name of the input file (file name without the extention). If the input was a directory containing pdfs, then a sub directory is created for each pdf file (named as the base name of the pdf files). 

The output directory structure is the following:

```
target_directory
  |__pdf_directory_1
  |  |__figures
  |  |  |_figure_1.png
  |  |  |_...
  |  |__tables
  |  |  |_table_1.csv
  |  |  |_...
  |  |_full_text.txt
  |__pdf_directory_2
     |_...
     ...
```
### II.2. Preprocessing
After the PDF extraction, preprocessing and deduplication can be applied prior to knowledge extraction. Currently, the proprocessing does very basic filtering using regex and isn't written to be extensible, although making it such is a perspective. 

The preprocessing can be run with the `python 2_preprocess.py input_path --corpus_file full_text.txt`, where:
 - `input_path` is either a signle text file, or a directory. If the latter, then the script expect the directory structure to be that produced by the extraction script. 
 - `--corpus_path full_text.txt` This optional parameter indicates the name of the corpus file in the directory structure, if the input was a directory. Default: `full_text.txt`

The script will modify the text files in-place. 

### II.3. Deduplication
Then one can run the deduplication script, that removes redundent text, notably recurring titles, numberings, etc. to make the input to the extraction cleaner. For now this feature is file-wise, but it is planned to add a corpus-wide option. 

Tu run deduplication, one may run `python 3_deduplicate.py` with the following syntax:
```usage: 3_deduplicate.py [-h] [--threshold THRESHOLD] [--num_perm NUM_PERM] [--ngram NGRAM] [--batch_size BATCH_SIZE] [--corpus_file CORPUS_FILE] path```

 - `--threshold VALUE` the similarity threshold to use
 - `--num_perm VALUE` number of permutations 
 - `--ngram VALUE` the maximum size of ngrams to use
 - `--batch_size VALUE` the batch size for the processing
 - `--corpus_file VALUE` if the input is a directory name, this parameter gives the name of the text file containing the extracted corpus. Default: full_text.txt
 - `path` the input path, if it is a file, deduplication is written in-place, of it is a directory, assumes it is structured in the way generated by the pdf extraction step. 
 - 
### II.4. Knowledge Extraction and RDF Generation
The knowledge extraction and KG generation script can be run with the following syntax: 
```
usage: 4_extract_terminology_and_relations.py [-h] [--extractors EXTRACTORS [EXTRACTORS ...]] [--endpoint ENDPOINT] [--corpus_file CORPUS_FILE] [--prefix-name PREFIX_NAME] [--prefix-uri PREFIX_URI] path

Knowledge Extractor

positional arguments:
  path                  Specify input extraction or input folder containing extractions

options:
  -h, --help            show this help message and exit
  --extractors EXTRACTORS [EXTRACTORS ...], -x EXTRACTORS [EXTRACTORS ...]
                        List of knowledge extractors to include. Possible: text2tcs, termsuite, entityfishing, spotlight, ncboannotator, usea. Default: text2tcs, termsuite.
  --corpus_file CORPUS_FILE
                        if the input is a directory name, this parameter gives the name of the text file containing the extracted corpus. Default: deduplicated.txt
  --prefix-name PREFIX_NAME
                        Prefix name for the model. Default: kext
  --prefix-uri PREFIX_URI
                        Prefix URI for the model. Default: http://w3id.org/kext/
```  

## III. Configuration and deployment of required APIs and tools
This section will give some instructions and recommendations on how to configure and set-up the various services supported by the framework.


### III.1. Termsuite 
Termsuite is normally available as a java library that must be integrated in a java programme or used through a command-line interface. To facilitate integration, we have developed a very basic REST API wrapper around term suite. 
The API version can be found here: https://github.com/EuromovDHM-SemTaxM/termsuite_rest_wrapper, you may run it by following the instructions. Once the API is running, termsuite can be used for knowledge extraction. 

You can find the configuration in `config/termsuite.json`, please change the `endpoint` parameter to "http://HOST:PORT/extract_terminology", replacing HOST and PORT as required. 

### III.2. Text2TCS 
Text2TCS is served through the European Language Grid, either through their online hosted API, or offline through a docker container. 
Currently, the supplied docker command for local deployment does sucessfully deploy the service, but the specifics of which route should be queried is not documented (not the same as the ELG online service). For our purposes, we used the online ELG API through the `elg` python package. The service is metered, but the allowance of 1500 daily calls is generous: We chunk the text to fit into the 7500 query limit, which means that even for a fairly long documents, the number of chunks is limited (~1 per page). Given the processing speed, it's unlikely to reach the limit within a day. 

Support for access through the API is a planned feature, but currently only accessible through the python wrapper. 
The only configuration parameter in `config/text2tcs.json` is 'elg_app_id`, which should be left as-is. There are no API keys or authentication tokens, as the ELG python API will interactively give a link to open to get a one-time token that must then be pasted on the standard input. This means that the extraction process cannot be scheduled through cron when including the text2tcx extractor, it must be started manually to paste the token. Once the token is pasted the authentication is cached and valid for a few hours. 

### III.3. Entity Fishing
(Entity Fishing)[https://github.com/kermitt2/entity-fishing] is a multilingual Entity Linker for Wikidata, while it isn't as accurate as SOTA models (e.g. BLINK), it is production ready, and ships with models for 15 languages. Entity-fishing also offers an Wikidata concept query API, which allows fast retrieval of relations and information about a particular concept without having to use SPARQL queries. Our component uses this API to retrieve relations between identified entities within the text (dicarding other relations).
While there is an online API, it's meant for demo purposes and the threshold for blacklisting IPs sending-out streams of queries fairly low. Which is why we recommend to deploy entity-fishing locally by following the procedure detailed here: https://nerd.readthedocs.io/en/latest/docker.html#running-entity-fishing-with-docker.

The configuration parameter located in `config/entityfishing.json` are the REST API endpoint URL: `endpoint` and the wikidata entity prefix URI `wikidata_prefix_uri`, which is useful to rebuild proper URIs after querying the concept index. The default endpoint is the online one at `http://nerd.huma-num.fr/nerd/service`.

### III.4. Spotlight
Spotlight can either be used through the online API or through locally deployed language-specific docker containers. The instructions to deploy over docker can be found here: https://github.com/dbpedia-spotlight/spotlight-docker
There are three configuration keys in the configuration file `config/spotlight.json`: 
 - `endpoint`: the REST API URL. For the official API: `https://api.dbpedia-spotlight.org/LANG/annotate` where LANG is the iso2 language code. 
 - `dbpedia_prefix_uri`: the dbpedia prefix for resources for the language of the documents being processed `http://LANG.dbpedia.org/resource/`, where LANG is the iso2 language code. 
 - `dbpedia_sparql_endpoint`: URI to a SPARQL endoint serving the corresponding language edition of spotlight. For hte official endpoint, use `https://fr.dbpedia.org/sparql`, where LANG is the iso2 language code. This is used to project relations unto extracted entities. 

### III.5. NCBO Annotator
The NCBO Annotator connector allows to interface with NCBO Bioportal Annotator or Annotator+ (https://bioportal.bioontology.org/) and any of the spin-off portals from the ontoportal alliance (https://ontoportal.org/). Paper: In the paper we used SIFR Bioportal, a french instance of Bioportal. 

The recommended usage is to point to the official APIs, which have a robust capacity for reasonable workloads (less than a few hours of uninterrupted queries). You will need to create an account on the target portal and get the API key authentication token from your user profile page. 
It is also possible to deploy a local instance with docker-compose or request a virtual machine from the Center for Biomedical Informatics at Stanford, however, you will have to load your own ontologies. Since many of the ontologies on the portals cannot be downloaded because of licensing restrictions,  making it difficult to replicate the same bredth. 

There are two configuration parameters in the configuration file: `config/ncboannotator.json`:
 - `endpoint`: the URL of the REST API of the targeted portal (e.g. `http://data.bioportal.lirmm.fr/annotator` for SIFR Bioportal). 
 - `apikey`: The API key obtained from the portal (even the docker or VM deployments have a default API key). 

### III.6. USEA
USEA is a system for joint, WSD, Semantic Role Labelling and Abstract Mearning Representation parsing by SapienzaNLP (https://github.com/SapienzaNLP/usea). USEA can be integrated through the provided Docker container. However, the official distribution contains errors in the source code in the docker container for the WSD component as well as bad exception handling, making them inoperable out-of-the-box.  Given that only images are available but not Dockerfiles, the errors must be solved interactively by running interactive shells inside the running containers (https://github.com/SapienzaNLP/usea/issues/4). Instructions pending.

## Full evaluation report from the paper for the considered use-case
```"DescriptiveStatisticsEvaluator": {
    "Termsuite REST": {
      "num_entities": 3092,
      "num_mentions": 31904,
      "num_mappings": 0,
      "avg_len": 12.46862871927555,
      "num_relations": 1369,
      "unique_sources": 531,
      "unique_targets": 787,
      "num_relation_types": 2
    },
    "Text2TCS ELG": {
      "num_entities": 10053,
      "num_mentions": 477783,
      "num_mappings": 0,
      "avg_len": 12.428429324579728,
      "num_relations": 28844,
      "unique_sources": 247,
      "unique_targets": 244,
      "num_relation_types": 9
    },
    "Entity Fishing": {
      "num_entities": 288,
      "num_mentions": 3558,
      "num_mappings": 0,
      "avg_len": 13.059027777777779,
      "num_relations": 281,
      "unique_sources": 130,
      "unique_targets": 59,
      "num_relation_types": 43
    },
    "Spotlight": {
      "num_entities": 572,
      "num_mentions": 4515,
      "num_mappings": 0,
      "avg_len": 15.183566433566433,
      "num_relations": 3484,
      "unique_sources": 484,
      "unique_targets": 351,
      "num_relation_types": 102
    },
    "NCBO Annotator": {
      "num_entities": 31829,
      "num_mentions": 54105,
      "num_mappings": 0,
      "avg_len": 7.534418297778755,
      "num_relations": 0,
      "unique_sources": 0,
      "unique_targets": 0,
      "num_relation_types": 0
    },
    "USEA Annotator": {
      "num_entities": 31354,
      "num_mentions": 33764,
      "num_mappings": 0,
      "avg_len": 12.0,
      "num_relations": 0,
      "unique_sources": 0,
      "unique_targets": 0,
      "num_relation_types": 0,
      "semantic_roles": 5255,
      "amr_graphs": 4591
    }
  },
  "OverlapEvaluator": {
    "Termsuite REST_vs_Text2TCS ELG": {
      "total_Termsuite REST": "3092",
      "total_Text2TCS ELG": "10053",
      "common (%Termsuite REST %Text2TCS ELG)": "1488 (0.48 0.15)",
      "common_partial_Termsuite REST_leq_Text2TCS ELG (%Text2TCS ELG)": "6771 (0.67)",
      "common_partial_Text2TCS ELG_leq_Termsuite REST (%Termsuite REST)": "618 (0.20)",
      "specific_Termsuite REST (%Termsuite REST)": "0 (0.00)",
      "specific_Text2TCS ELG (%Text2TCS ELG)": "1176 (0.12)",
      "relations with shared source+target (%Termsuite REST %Text2TCS ELG)": "0 (0.00 0.00)"
    },
    "Termsuite REST_vs_Entity Fishing": {
      "total_Termsuite REST": "3092",
      "total_Entity Fishing": "288",
      "common (%Termsuite REST %Entity Fishing)": "155 (0.05 0.54)",
      "common_partial_Termsuite REST_leq_Entity Fishing (%Entity Fishing)": "39 (0.14)",
      "common_partial_Entity Fishing_leq_Termsuite REST (%Termsuite REST)": "91 (0.03)",
      "specific_Termsuite REST (%Termsuite REST)": "2807 (0.91)",
      "specific_Entity Fishing (%Entity Fishing)": "3 (0.01)",
      "relations with shared source+target (%Termsuite REST %Entity Fishing)": "0 (0.00 0.00)"
    },
    "Termsuite REST_vs_Spotlight": {
      "total_Termsuite REST": "3092",
      "total_Spotlight": "572",
      "common (%Termsuite REST %Spotlight)": "254 (0.08 0.44)",
      "common_partial_Termsuite REST_leq_Spotlight (%Spotlight)": "76 (0.13)",
      "common_partial_Spotlight_leq_Termsuite REST (%Termsuite REST)": "57 (0.02)",
      "specific_Termsuite REST (%Termsuite REST)": "2705 (0.87)",
      "specific_Spotlight (%Spotlight)": "185 (0.32)",
      "relations with shared source+target (%Termsuite REST %Spotlight)": "0 (0.00 0.00)"
    },
    "Termsuite REST_vs_NCBO Annotator": {
      "total_Termsuite REST": "3092",
      "total_NCBO Annotator": "31829",
      "common (%Termsuite REST %NCBO Annotator)": "3 (0.00 0.00)",
      "common_partial_Termsuite REST_leq_NCBO Annotator (%NCBO Annotator)": "0 (0.00)",
      "common_partial_NCBO Annotator_leq_Termsuite REST (%Termsuite REST)": "6 (0.00)",
      "specific_Termsuite REST (%Termsuite REST)": "3083 (1.00)",
      "specific_NCBO Annotator (%NCBO Annotator)": "31820 (1.00)"
    },
    "Termsuite REST_vs_USEA Annotator": {
      "total_Termsuite REST": "3092",
      "total_USEA Annotator": "31354",
      "common (%Termsuite REST %USEA Annotator)": "2213 (0.72 0.07)",
      "common_partial_Termsuite REST_leq_USEA Annotator (%USEA Annotator)": "837 (0.03)",
      "common_partial_USEA Annotator_leq_Termsuite REST (%Termsuite REST)": "611 (0.20)",
      "specific_Termsuite REST (%Termsuite REST)": "0 (0.00)",
      "specific_USEA Annotator (%USEA Annotator)": "27693 (0.88)"
    },
    "Text2TCS ELG_vs_Entity Fishing": {
      "total_Text2TCS ELG": "10053",
      "total_Entity Fishing": "288",
      "common (%Text2TCS ELG %Entity Fishing)": "252 (0.03 0.88)",
      "common_partial_Text2TCS ELG_leq_Entity Fishing (%Entity Fishing)": "50 (0.17)",
      "common_partial_Entity Fishing_leq_Text2TCS ELG (%Text2TCS ELG)": "1091 (0.11)",
      "specific_Text2TCS ELG (%Text2TCS ELG)": "8660 (0.86)",
      "specific_Entity Fishing (%Entity Fishing)": "0 (0.00)",
      "relations with shared source+target (%Text2TCS ELG %Entity Fishing)": "0 (0.00 0.00)"
    },
    "Text2TCS ELG_vs_Spotlight": {
      "total_Text2TCS ELG": "10053",
      "total_Spotlight": "572",
      "common (%Text2TCS ELG %Spotlight)": "307 (0.03 0.54)",
      "common_partial_Text2TCS ELG_leq_Spotlight (%Spotlight)": "83 (0.15)",
      "common_partial_Spotlight_leq_Text2TCS ELG (%Text2TCS ELG)": "360 (0.04)",
      "specific_Text2TCS ELG (%Text2TCS ELG)": "9303 (0.93)",
      "specific_Spotlight (%Spotlight)": "0 (0.00)",
      "relations with shared source+target (%Text2TCS ELG %Spotlight)": "0 (0.00 0.00)"
    },
    "Text2TCS ELG_vs_NCBO Annotator": {
      "total_Text2TCS ELG": "10053",
      "total_NCBO Annotator": "31829",
      "common (%Text2TCS ELG %NCBO Annotator)": "6 (0.00 0.00)",
      "common_partial_Text2TCS ELG_leq_NCBO Annotator (%NCBO Annotator)": "0 (0.00)",
      "common_partial_NCBO Annotator_leq_Text2TCS ELG (%Text2TCS ELG)": "146 (0.01)",
      "specific_Text2TCS ELG (%Text2TCS ELG)": "9901 (0.98)",
      "specific_NCBO Annotator (%NCBO Annotator)": "31677 (1.00)"
    },
    "Text2TCS ELG_vs_USEA Annotator": {
      "total_Text2TCS ELG": "10053",
      "total_USEA Annotator": "31354",
      "common (%Text2TCS ELG %USEA Annotator)": "3077 (0.31 0.10)",
      "common_partial_Text2TCS ELG_leq_USEA Annotator (%USEA Annotator)": "865 (0.03)",
      "common_partial_USEA Annotator_leq_Text2TCS ELG (%Text2TCS ELG)": "1701 (0.17)",
      "specific_Text2TCS ELG (%Text2TCS ELG)": "4410 (0.44)",
      "specific_USEA Annotator (%USEA Annotator)": "25711 (0.82)"
    },
    "Entity Fishing_vs_Spotlight": {
      "total_Entity Fishing": "288",
      "total_Spotlight": "572",
      "common (%Entity Fishing %Spotlight)": "168 (0.58 0.29)",
      "common_partial_Entity Fishing_leq_Spotlight (%Spotlight)": "10 (0.02)",
      "common_partial_Spotlight_leq_Entity Fishing (%Entity Fishing)": "3 (0.01)",
      "specific_Entity Fishing (%Entity Fishing)": "107 (0.37)",
      "specific_Spotlight (%Spotlight)": "391 (0.68)",
      "relations with shared source+target (%Entity Fishing %Spotlight)": "13 (0.05 0.00)"
    },
    "Entity Fishing_vs_NCBO Annotator": {
      "total_Entity Fishing": "288",
      "total_NCBO Annotator": "31829",
      "common (%Entity Fishing %NCBO Annotator)": "0 (0.00 0.00)",
      "common_partial_Entity Fishing_leq_NCBO Annotator (%NCBO Annotator)": "0 (0.00)",
      "common_partial_NCBO Annotator_leq_Entity Fishing (%Entity Fishing)": "0 (0.00)",
      "specific_Entity Fishing (%Entity Fishing)": "288 (1.00)",
      "specific_NCBO Annotator (%NCBO Annotator)": "31829 (1.00)"
    },
    "Entity Fishing_vs_USEA Annotator": {
      "total_Entity Fishing": "288",
      "total_USEA Annotator": "31354",
      "common (%Entity Fishing %USEA Annotator)": "247 (0.86 0.01)",
      "common_partial_Entity Fishing_leq_USEA Annotator (%USEA Annotator)": "172 (0.01)",
      "common_partial_USEA Annotator_leq_Entity Fishing (%Entity Fishing)": "4 (0.01)",
      "specific_Entity Fishing (%Entity Fishing)": "0 (0.00)",
      "specific_USEA Annotator (%USEA Annotator)": "30931 (0.99)"
    },
    "Spotlight_vs_NCBO Annotator": {
      "total_Spotlight": "572",
      "total_NCBO Annotator": "31829",
      "common (%Spotlight %NCBO Annotator)": "0 (0.00 0.00)",
      "common_partial_Spotlight_leq_NCBO Annotator (%NCBO Annotator)": "0 (0.00)",
      "common_partial_NCBO Annotator_leq_Spotlight (%Spotlight)": "0 (0.00)",
      "specific_Spotlight (%Spotlight)": "572 (1.00)",
      "specific_NCBO Annotator (%NCBO Annotator)": "31829 (1.00)"
    },
    "Spotlight_vs_USEA Annotator": {
      "total_Spotlight": "572",
      "total_USEA Annotator": "31354",
      "common (%Spotlight %USEA Annotator)": "414 (0.72 0.01)",
      "common_partial_Spotlight_leq_USEA Annotator (%USEA Annotator)": "90 (0.00)",
      "common_partial_USEA Annotator_leq_Spotlight (%Spotlight)": "3 (0.01)",
      "specific_Spotlight (%Spotlight)": "65 (0.11)",
      "specific_USEA Annotator (%USEA Annotator)": "30847 (0.98)"
    },
    "NCBO Annotator_vs_USEA Annotator": {
      "total_NCBO Annotator": "31829",
      "total_USEA Annotator": "31354",
      "common (%NCBO Annotator %USEA Annotator)": "0 (0.00 0.00)",
      "common_partial_NCBO Annotator_leq_USEA Annotator (%USEA Annotator)": "0 (0.00)",
      "common_partial_USEA Annotator_leq_NCBO Annotator (%NCBO Annotator)": "0 (0.00)",
      "specific_NCBO Annotator (%NCBO Annotator)": "31829 (1.00)",
      "specific_USEA Annotator (%USEA Annotator)": "31354 (1.00)"
    }
  }
}```
