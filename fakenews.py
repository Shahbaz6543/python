import random
subjects = [
    "ShahyesRukh Khan",
    "Virat Kohli",
    "Prime Minister Modi",
    "Indian Government",
    "ISRO Scientists",
    "Supreme Court",
    "Indian Army",
    "A group of monkeys",
    "Bollywood celebrities",
    "Tech startups in India",
    "A mysterious man",
    "Indian students",
    "A famous YouTuber"
]

actions = [
    "launches",
    "announces",
    "cancels",
    "reveals",
    "wins",
    "loses",
    "surprises everyone by",
    "shocks the nation by",
    "celebrates",
    "tests",
    "bans",
    "approves"
]

places_or_things = [
    "at Red Fort",
    "in Mumbai",
    "during a live event",
    "inside Parliament",
    "on social media",
    "at midnight",
    "during IPL match",
    "for the first time",
    "across the country",
    "in a press conference",
    "in New Delhi",
    "on national television"
]

while True:
    subject=random.choice(subjects)
    action=random.choice(actions)
    places_or_thing=random.choice(places_or_things)

    headline=f" BREAKING NEWS: {subject} {action} {places_or_thing} "
    print("\n"+headline) 

    user_input=input("\n Do you want another headline? (yes/no)").strip().lower()
    if user_input=="no" :
        break 
print("thanks")  