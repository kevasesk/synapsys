window.Slider = function () {
    return {
        slides: [
            {
                img: '/films/avatar.jpg',
                title: 'Avatar (2009)',
                description: 'A paraplegic Marine dispatched to the moon Pandora on a unique mission becomes torn between following his orders and protecting the world he feels is his home.'
            },
            {
                img: '/films/avangers.jpg',
                title: 'Avengers: Endgame (2019)',
                description: 'Following the devastating events of &quot;Avengers: Infinity War,&quot; the remaining heroes, scattered and defeated, must assemble once more to reverse the actions of the mad titan Thanos and restore balance to the universe, no matter the personal cost.'
            },
            {
                img: '/films/titanic.jpg',
                title: 'Titanic (1997)',
                description: 'A seventeen-year-old aristocrat, engaged to a wealthy suitor, falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic. Their forbidden romance is set against the backdrop of the ship\'s tragic maiden voyage.'
            }
        ],
        currentIndex: 0,
        next() {
            this.currentIndex = (this.currentIndex + 1) % this.slides.length;
        },
        prev() {
            this.currentIndex = (this.currentIndex - 1 + this.slides.length) % this.slides.length;
        },
        goTo(index) {
            this.currentIndex = index;
        }
    }
};