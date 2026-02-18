<div align="center">
  <h2 align="center">Places Scout</h2>
  <div align="left">

![Repo Views](https://visitor-badge.laobi.icu/badge?page_id=SpencerVJones/PlacesScout)

</div>

  
<p align="center">
  A global relocation intelligence app for finding cities that match your moving priorities.  
  Built with <strong>Python, Streamlit, Folium, and Pandas</strong>, using <strong>GeoNames + World Bank</strong> public datasets for transparent multi-factor city scoring.
  <br /><br />
  This project focuses on practical decision support: tune your constraints, adjust scoring weights, explore interactive map results, and export shortlists.
  <br />
  <br />
  <a href="https://github.com/SpencerVJones/PlacesScout/issues">Report Bug</a>
    ·
    <a href="https://github.com/SpencerVJones/PlacesScout/issues">Request Feature</a>
  </p>
</div>


<!-- PROJECT SHIELDS -->
<div align="center">


![License](https://img.shields.io/badge/License-Proprietary-black?style=for-the-badge)
![Contributors](https://img.shields.io/github/contributors/SpencerVJones/PlacesScout?style=for-the-badge)
![Forks](https://img.shields.io/github/forks/SpencerVJones/PlacesScout?style=for-the-badge)
![Stargazers](https://img.shields.io/github/stars/SpencerVJones/PlacesScout?style=for-the-badge)
![Issues](https://img.shields.io/github/issues/SpencerVJones/PlacesScout?style=for-the-badge)
![Last Commit](https://img.shields.io/github/last-commit/SpencerVJones/PlacesScout?style=for-the-badge)
![Repo Size](https://img.shields.io/github/repo-size/SpencerVJones/PlacesScout?style=for-the-badge)

![Platform](https://img.shields.io/badge/Platform-Web-lightgrey.svg?style=for-the-badge&logo=google-chrome&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-E34F26?style=for-the-badge&logo=streamlit&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-Map%20Rendering-3D9970?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)
![Open Data](https://img.shields.io/badge/Open%20Data-GeoNames%20%2B%20World%20Bank-4C9A2A?style=for-the-badge)


</div>



## 📑 Table of Contents
- [📑 Table of Contents](#-table-of-contents)
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Architecture](#architecture)
- [Features](#features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [How to Run](#how-to-run)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributors](#contributors)
- [License](#license)
- [Contact](#contact)

## Overview
**Places Scout** is a global city scoring and exploration tool for relocation decisions.  
It combines public datasets with a weighted scoring model so you can compare cities by affordability, safety, transit, and amenities.

This repository is structured as a **Python app + CLI workflow**, with Streamlit for interactive UI and Folium for map rendering.  


## Technologies Used
- **Python 3**
- **Streamlit**
- **Folium**
- **Pandas**
- **Requests**
- **GitHub Actions** (CI)

## Architecture
- **Streamlit web app** (`app.py`) for interactive filtering, shortlist export, and map display  
- **Global dataset/scoring pipeline** (`major_cities.py`) for fetching, enrichment, and ranking  
- **CLI map generator** (`generate_all_maps.py`) for static HTML map output  
- Shared config and validation modules for **weights** and **cache schema**

## Features
- 🌍 Global city coverage using GeoNames `cities500`
- ⚖️ Weighted multi-factor scoring (affordability, safety, transit, amenities)
- 🎛️ Real-time moving filters (cost, crime, quality-of-life, LGBT equality, beach/mountain)
- 🗺️ Interactive Folium map embedded in Streamlit
- 📥 CSV shortlist export for offline comparison
- 🤖 CI pipeline with lint + automated unit tests

## Demo
🔗 **Live:** Coming Soon

Run locally using the commands below.


## Project Structure
```bash
Folium-Web-map/
├── .github/workflows/ci.yml      # CI pipeline (lint + tests)
├── data/                          # Cached datasets
├── tests/                         # Unit tests
├── app.py                         # Streamlit app
├── generate_all_maps.py           # CLI map generation entry point
├── major_cities.py                # Global data fetch/enrichment/scoring/map builder
├── scoring_config.py              # Weight defaults + weight parser
├── world_cache_schema.py          # Global cache schema validator
├── requirements.txt               # Python dependencies
└── README.md
```
## Testing
```bash
python3 -m unittest discover -s tests -v
```


## Getting Started
### Prerequisites
- **Python 3.11+**
- `pip`

### Installation
```bash
git clone https://github.com/SpencerVJones/Folium-Web-map.git
cd Folium-Web-map
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```
 
### How to Run
```bash
python3 -m streamlit run app.py
```

Optional CLI map build:
```bash
python3 generate_all_maps.py --min-population 50000 --output world_cities_map.html
```

## Usage
- Open the Streamlit URL shown in your terminal.
- Tune weights and filters based on your moving priorities.
- Review map + shortlist results and export CSV if needed.
 
## Roadmap
 - [x] Add deployable public demo (Streamlit Cloud)
 - [ ] Expand test coverage for data fallback + sorting edge cases
 - [ ] Add richer comparison/insight views in-app

See open issues for a full list of proposed features (and known issues).
 

## Contributors
<a href="https://github.com/SpencerVJones/PlacesScout/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SpencerVJones/PlacesScout"/>
</a>


## License
Copyright (c) 2026 Spencer Jones
<br>
All rights reserved.
<br>
Permission is granted to view this code for personal and educational purposes only.
<br>
No permission is granted to copy, modify, distribute, sublicense, or sell any portion of this code without explicit written consent from the author.


## Contact
Spencer Jones
📧 [SpencerVJones@outlook.com](mailto:SpencerVJones@outlook.com)  
🔗 [GitHub Profile](https://github.com/SpencerVJones)  
🔗 [Project Repository](https://github.com/SpencerVJones/PlacesScout)
