import random

def sample_population(event_list, weighted_labels):
    """ Returns a list containing one tuple (sample, label) for each of the lists in event_list.
        The samples are selected so the proportion of associated labels in the final result
        respects the description in weighted_labels.

        @param event_list a list of lists, each sublist containing tuples (event, label)
        @param weighted_labels a list of tuples (label, weight)
    """
    selected_events = []
    used_labels = []
    pending_event_list = []

    # Randomly suffle event_list to avoid any bias due to the order of the list
    shuffled_event_list = event_list[:]
    random.shuffle(shuffled_event_list)

    # Select random events from each of the sublists until an event from each sublist is selected
    while len(selected_events) < len(event_list):
        # Remove first sublist and shuffle it
        event_sublist = shuffled_event_list.pop(0)
        shuffled_event_sublist = event_sublist[:]
        random.shuffle(shuffled_event_sublist)

        is_event_found = False

        # Remove events from the sublist until we find an event whose label is not used yet
        while shuffled_event_sublist:
            random_event, random_label = shuffled_event_sublist.pop(0)

            if random_label not in used_labels:
                selected_events.append((random_event, random_label))
                used_labels.append(random_label)
                shuffled_event_sublist = None
                is_event_found = True

        # If we couldn't find an event for the current sublist (because all the labels of events
        # have already been used), we set the sublist as pending. We'll try again later in a
        # second iteration.
        if not is_event_found:
            pending_event_list.append(event_sublist)

        # If we emptied the event list, we reset the used labels and restart again with the
        # pending sublists
        if not shuffled_event_list:
            used_labels = []
            shuffled_event_list = pending_event_list
            pending_event_list = []

    return selected_events


if __name__ == "__main__":
    import itertools
    import bisect

    def sample_labels(weighted_labels):
        """ Provides with an iterator that returns a random label in each call. The probability
            of returning a specific label is defined by the weighted_labels parameter."""
        # Based on example of Python documentation. First, build list of labels and their
        # associated weights
        labels, weights = zip(*weighted_labels)

        # Compute list of accumulated weights
        cumdist = list(itertools.accumulate(weights))

        # Compute random sample
        while True:
            proportion = random.random() * cumdist[-1]
            yield labels[bisect.bisect(cumdist, proportion)]

    # Build weighted labels
    weighted_labels = [('A', 50), ('B', 22), ('C', 17), ('D', 1)]
    print("Weighted labels: {}\n".format(weighted_labels))

    # Build population
    weighted_iter = sample_labels(weighted_labels)
    num_lists = 10
    num_elems = 6

    event_list = [[(10*j + i, next(weighted_iter)) for i in range(num_elems)]
                                                   for j in range(num_lists)]

    for event_sublist_index, event_sublist in enumerate(event_list):
        print("List {}: {}".format(event_sublist_index, event_sublist))
    print()

    # Compute and print selection of events
    selected_events = sample_population(event_list, weighted_labels)
    print("Selected events: {}\n".format(sorted(selected_events)))

    # Print proportion of labels for the selected events
    for label, weight in weighted_labels:
        count = len(list(filter(lambda x: x[1] == label, selected_events)))
        print("Proportion of label {}: {:.2f}".format(label, count / len(selected_events)))

