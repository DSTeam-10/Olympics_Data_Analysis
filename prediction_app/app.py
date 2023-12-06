from flask import Flask, render_template, request
import pickle
import pandas as pd


app = Flask(__name__)

# Load the KMeans model
with open('kmeans_model.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

# Load the StandardScaler model
with open('scaler_model.pkl', 'rb') as file:
    scaler_model = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    df = pd.read_csv("../output.csv")
    if request.method == 'POST':
        team_name = request.form.get('teamName').capitalize()
        # if team_name in dataset_name:
        #     get the values of gold medal cloumn, silver medal column, bronze medal coulmn and store it in 3 variables

    # Assuming df is your DataFrame and team_name is the specific team name you are interested in
    # team_name = "YourTeamName"

    # Check if the team_name is in the DataFrame
        if team_name in df['Team'].values:
            # Extract values for gold, silver, and bronze columns for the specified team
            gold_medal = int(df.loc[df['Team'] ==
                                    team_name, 'Total_Gold_Medals'].values)
            silver_medal = int(df.loc[df['Team'] ==
                                      team_name, 'Total_Silver_Medals'].values)
            bronze_medal = int(df.loc[df['Team'] ==
                                      team_name, 'Total_Bronze_Medals'].values)

            # Print or use the extracted values as needed
            print(f"Gold values for {team_name}: {gold_medal}")
            print(f"Silver values for {team_name}: {silver_medal}")
            print(f"Bronze values for {team_name}: {bronze_medal}")

            total_medal_value = calculate_total_medal_value(
                gold_medal, silver_medal, bronze_medal)
            print(total_medal_value)

            # arr1 = np.array([team_name, team_medal])
            # pred = model.predict([final_arr])
            # Preprocess the input data using the loaded scaler
            scaled_data = scaler_model.transform([[total_medal_value]])

            # # Make predictions using the loaded KMeans model
            cluster_prediction = kmeans_model.predict(scaled_data)[0]
            cluster_labeling = cluster_labels(cluster_prediction)
            print(cluster_prediction, cluster_labeling)

            data = {"team_name": team_name,
                    "gold_medal": gold_medal,
                    "silver_medal": silver_medal,
                    "bronze_medal": bronze_medal,
                    "prediction": cluster_labeling}

            return render_template("index.html", data=data)
        else:
            return render_template("index.html", msg=f"Team {team_name} not found in the dataset.")

        # You can now use 'cluster_prediction' in your application logic as needed
        # For example, you might want to display it in the template

    return render_template("index.html")


def calculate_total_medal_value(*args):
    return args[0] * 3 + args[1] * 2 + args[2] * 1


def cluster_labels(cluster):
    label_mapping = {0: 'low', 1: 'high', 2: 'medium'}

    return label_mapping.get(cluster, 'unknown')


if __name__ == '__main__':
    app.run(debug=True)
