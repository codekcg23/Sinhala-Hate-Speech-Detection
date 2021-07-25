import neptune.new as neptune

run = neptune.init(project='codekcg23/Research-Experiments',
                   api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZmJhODVhMi1mYzI0LTQ5MjMtOTExOC00YzMzMmI1YTdjZmIifQ==')  # your credentials

run["JIRA"] = "NPT-952"
run["parameters"] = {"learning_rate": 0.001,
                     "optimizer": "Adam"}

for epoch in range(100):
    run["train/loss"].log(epoch * 0.4)
run["eval/f1_score"] = 0.66
