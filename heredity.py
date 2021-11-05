import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # jprobability holds the joint probability
    jprobability = 1.0
    for person in people:

        #num_genes holds the number of genes the person has.
        num_genes = 0

        if person in one_gene:
            num_genes = 1
        elif person in two_genes:
            num_genes = 2

        if people[person]["mother"] is not None and people[person]["father"] is not None:
            # parents holds the names of the person's parents
            parents = [people[person]["mother"], people[person]["father"]]
            # parent_prob holds the parent's probabilities of passing their genes
            parent_prob = [0.0, 0.0]
        else:
            parents = [None]
            # if the person does not have parents then we calculate their probability using PROBS.
            jprobability *= PROBS["gene"][num_genes]

        if parents[0] is not None:
            for num in range(2):
                # if the parent is in two_genes then they have a 100% chance of passing the gene unless it mutates.
                if parents[num] in two_genes:
                    parent_prob[num] = 1 - PROBS["mutation"]
                # if the parent is one_gene, then they have a 50% chance of passing the gene.
                elif parents[num] in one_gene:
                    parent_prob[num] = 0.5
                # if the parent doesn't have any genes, then they won't pass it unless it mutates.
                else:
                    parent_prob[num] = PROBS["mutation"]

            if num_genes == 0:
                # if the person does not have genes, then we calculate the probability that neither of their parents
                # passed the gene
                jprobability *= (1-parent_prob[0]) * (1-parent_prob[1])
            elif num_genes == 1:
                # if the person has one gene, then we calculate the probability that they get the gene from one parent
                # or the other one.
                jprobability *= (parent_prob[0] * (1-parent_prob[1])) + ((1-parent_prob[0]) * parent_prob[1])
            else:
                # if the person has 2 genes, then we calculate the probability that they got a gene from both parents.
                jprobability *= (parent_prob[0] * parent_prob[1])

        if person in have_trait:
            # if the person has the trait then we calculate the probability of them having the gene using PROBS
            jprobability *= PROBS["trait"][num_genes][True]
        else:
            # if the person does not have the trait then we calculate the probability of them having the gene using
            # PROBS
            jprobability *= PROBS["trait"][num_genes][False]

    return jprobability


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        geneholder = (
            1 if person in one_gene else
            2 if person in two_genes else
            0
        )
        if person in have_trait:
            trait = True
        else:
            trait = False
        probabilities[person]["trait"][trait] += p
        probabilities[person]["gene"][geneholder] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """

    for person in probabilities:
        # prob_value holds the sum of every probability
        prob_value = 0
        for num in range(3):
            prob_value += probabilities[person]["gene"][num]

    # We check to see if the values in the 'gene' key add up to one. If they do not add up to one, then we
    # multiply everything inside of 'gene' by one divided by the sum of its values to make it equal to one
        if prob_value != 1:
            for gene in probabilities[person]["gene"]:
                probabilities[person]["gene"][gene] *= (1/prob_value)

        prob_value = 0

        prob_value += probabilities[person]["trait"][True]
        prob_value += probabilities[person]["trait"][False]

        if prob_value != 1:
            probabilities[person]["trait"][True] *= (1/prob_value)
            probabilities[person]["trait"][False] *= (1/prob_value)


if __name__ == "__main__":
    main()
