# 🛡️ EcoRisk NJ: Cancer & Community Air Toxics Explorer

**EcoRisk NJ** is an advanced spatial mapping and environmental justice web application built for the **Congressional App Challenge**. It translates complex U.S. EPA toxicological models and geographic information systems (GIS) data into an interactive, real-time civic tool designed to empower New Jersey residents and policymakers.

---

## 🌟 Key Features

* **Interactive Air Toxics Risk Model (Tab 1):** Visualizes 2020 EPA Census Block excess lifetime cancer risk baselines for Ethylene Oxide ($EtO$) and industrial air toxics across New Jersey, complete with custom fenceline proximity screening.
* **Multi-Year Regional Exposure Explorer (Tab 2):** Tracks multi-year cross-border industrial emission trends across the New Jersey and East Coast airshed corridor.
* **Analytics & Trends (Tab 3):** Tracks multi-year cross-border industrial emission trends across the New Jersey and East Coast airshed corridor.
* **About EcoRisk NJ (Tab 4):** Scientific Methodology about Ethylene Oxide and its link to cancer.
* **Meet the Developer (Tab 5):** Meet the Developer and the motivation behind creating EcoRisk NJ.
* **Research Paper (Tab 6):** Official Research Paper & Documentation.

---

## 🛠️ Technology Stack

* **Programming Language:** Python
* **Web Framework:** Streamlit
* **Spatial & GIS Libraries:** GeoPandas, Shapely, Folium, Streamlit-Folium
* **Data Visualization:** Altair, Custom SaaS CSS UI components

---

## 🚀 Quick Start / Local Installation

To run EcoRisk NJ locally on **Windows**, follow these steps. Make sure Git and Python are installed first.

### Step 1: Clone the repository

Open **Command Prompt (CMD)** and run the following command. Replace `your-username` with the GitHub account that hosts this repository.

```cmd
git clone https://github.com/your-username/ecorisk-nj.git
```

### Step 2: Navigate to the project folder

In the same Command Prompt window, navigate to the cloned repository. Replace `<Your folder path>` with the actual location where you cloned it.

```cmd
cd "C:\<Your folder path>\ecorisk-nj"
```

### Step 3: Install dependencies

The repository includes a `requirements.txt` file listing the Python packages needed to run EcoRisk NJ. Install them by running:

```cmd
python -m pip install -r requirements.txt
```

### Step 4: Launch EcoRisk NJ

Run the following command in the same Command Prompt window:

```cmd
python -m streamlit run app.py
```

The app should open in your web browser. If it does not, open the local URL shown in Command Prompt.
