# Interactive Map with Folium
  <div align="left">
	
![Repo Views](https://visitor-badge.laobi.icu/badge?page_id=SpencerVJones/Folium-Web-map)
</div>
This Python script utilizes the Folium library to create an interactive map visualizing places where the author has lived and visited. It generates an HTML file with embedded JavaScript code to render the map.

## Features:
- **Base Map:** Utilizes the CartoDB Positron base map for visualization.
- **Marker Clusters:** Groups multiple markers at the same location to prevent overlap and improve map readability.
- **Marker Colors:** Distinguishes between places the author has lived (blue) and visited (green).
- **Pop-up Information:** Provides location-specific information when clicking on markers, such as city names.

## Installation:
To install Folium, use pip in the terminal:
```
pip3 install folium
```

## Usage:
- Ensure you have installed Folium (see Installation).
- Run the script in a Python environment.
- Open the generated HTML file (`map1.html`) in a web browser to view the interactive map.

## File Structure:
- `map1.html`: The output HTML file containing the interactive map.
- `Python Script`: The Python script (map_script.py) generates the map and saves it as an HTML file.


## Technologies Used:
- **Folium:** A Python library used for visualizing geospatial data. It leverages the `Leaflet.js` library to generate interactive maps.
- **Python:** The programming language used to write the script.
- **IDE:** PyCharm


## Contributing
Contributions are welcome! 

### You can contribute by:
-  Reporting bugs
-  Suggesting new features
-  Submitting pull requests to improve the codebase
