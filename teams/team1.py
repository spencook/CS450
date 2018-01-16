class Movie:
    def __init__(self):
        self.title = ""
        self.year = 0
        self.runtime = 0

    def calc_time(self):
        self.hours = self.runtime % 60

def main():
    movie = Movie()
    movie.title = "jurassic"
    movie.year = 2000
    movie.runtime = 20

    print(movie)

if __name__ == "__main__":
    main()
