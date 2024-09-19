from components.main import main

# Define the specific combinations you want
specific_combinations = [
    {"BATCH_SIZE": 1, "LEARNING_RATE": 2e-4, "TRAIN_NUMBER": "00"},
]

# Call main with the specific combinations
for params in specific_combinations:
    main(
        TRAIN_NUMBER=params["TRAIN_NUMBER"],
        BATCH_SIZE=params["BATCH_SIZE"],
        LEARNING_RATE=params["LEARNING_RATE"],
        EPOCH=2,
        BETA_1=0.5,
        ALPHA=0.1,
        BETA=0.1,
        DIS="v1",
        GEN="v1"
    )
