# also change explain method, shares logic
def calculate_score(self, user_interests, item_features, verbose=False):
    score = [[], [], ]
    all_positive_interests = []
    all_item_features = []

    for interest_rating, interests in user_interests.items():
        if len(interests) == 0 or interest_rating < 4:
            continue
        for interest in interests:
            if self.can_use_words_in_explanation(interest):
                all_positive_interests.append(interest)

    for _, features in item_features.items():
        for feature in features:
            if self.can_use_words_in_explanation(feature):
                all_item_features.append(feature)

    dists = []

    for feature in all_item_features:
        for interest in all_positive_interests:
            pair_distance = self._calculate_distance(interest, feature)
            dists.append(pair_distance)

    print(dists)

    """
    for i in range(6):
        score[i].sort()
        score[i] = np.mean((score[i] + [1, 1])[:2])
    """
    score[1] = np.mean(self.find_two_smallest(dists))

    return score