const sliderContainer = document.getElementById('slider-container');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const sliderDots = document.getElementById('slider-dots');

if (sliderContainer) { // Check if slider elements exist to avoid errors
    let currentIndex = 0;
    const films = sliderContainer.children;
    const totalFilms = films.length;

    function updateSlider() {
        sliderContainer.style.transform = `translateX(-${currentIndex * 100}%)`;
        updateDots();
    }

    function updateDots() {
        sliderDots.innerHTML = '';
        for (let i = 0; i < totalFilms; i++) {
            const dot = document.createElement('button');
            dot.className = `w-3 h-3 rounded-full ${i === currentIndex ? 'bg-blue-500' : 'bg-gray-400'}`;
            dot.addEventListener('click', () => {
                currentIndex = i;
                updateSlider();
            });
            sliderDots.appendChild(dot);
        }
    }

    prevBtn.addEventListener('click', () => {
        currentIndex = (currentIndex - 1 + totalFilms) % totalFilms;
        updateSlider();
    });

    nextBtn.addEventListener('click', () => {
        currentIndex = (currentIndex + 1) % totalFilms;
        updateSlider();
    });

    updateSlider();
}