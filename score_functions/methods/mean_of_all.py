def calculate_score(self, user_interests, item_features, verbose=False):
    score = [[], [], ]
    # mean_iidf = TODO should I really delete the following: (1 / self.idf_mean_review) * self.INVERSE_IDF_SCALING_CONSTANT
    for interest_rating, interests in user_interests.items():
        if len(interests) == 0 or interest_rating < 4:
            continue
        for interest in interests:
            dists = []
            for feature_rating, features in item_features.items():
                for feature in features:
                    pair_distance = self._calculate_distance(interest, feature)
                    if verbose:
                        print(f'({interest:<12}, {feature:<12}, {interest_rating}) '
                              f'-> '
                              f'pair_distance={pair_distance:<5}')
                    dists.append(pair_distance)
            if len(dists) > 0:
                distance = np.mean(dists)
                score[1].append(distance)

    """
    for i in range(6):
        score[i].sort()
        score[i] = np.mean((score[i] + [1, 1])[:2])
    """
    score[1] = np.mean(score[1]) if len(score[1]) else 1

    return score