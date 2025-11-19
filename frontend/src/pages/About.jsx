import React from 'react';

const impactHighlights = [
  {
    title: 'Farmer First',
    description: 'Give growers instant clarity on what is happening to their crops so they can act in hours instead of days.',
  },
  {
    title: 'Actionable Insights',
    description: 'Pair each prediction with recommended remedies, agronomy notes, and urgency indicators.',
  },
  {
    title: 'Continuous Learning',
    description: 'Every labeled upload helps retrain the model and expand coverage for new regions and crop varieties.',
  },
];

const About = () => {
  return (
    <div className="max-w-5xl mx-auto space-y-10 animate-fadeIn">
      <section className="text-center space-y-4">
        <p className="text-sm uppercase tracking-widest text-primary-600 font-semibold">
          About PlantDoc
        </p>
        <h1 className="text-4xl md:text-5xl font-extrabold text-gray-900">
          AI that protects every leaf
        </h1>
        <p className="text-lg text-gray-600 max-w-3xl mx-auto">
          PlantDoc combines field knowledge with computer vision to detect 38 plant diseases
          within seconds. The platform helps farmers, agronomists, and researchers make confident
          decisions without relying on expensive lab work or slow manual inspections.
        </p>
      </section>

      <section className="grid md:grid-cols-2 gap-6">
        <div className="card border-l-4 border-primary-500 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 mb-3">Mission</h2>
          <p className="text-gray-700 leading-relaxed">
            Accelerate sustainable farming by making crop-health intelligence accessible to every
            smartphone. We believe preventing outbreaks early is the fastest path to better yields,
            reduced chemical overuse, and healthier food systems.
          </p>
        </div>
        <div className="card border-l-4 border-emerald-500 shadow-lg">
          <h2 className="text-2xl font-bold text-gray-900 mb-3">How it works</h2>
          <p className="text-gray-700 leading-relaxed">
            Images are analyzed with a ResNet50 backbone that was fine-tuned on thousands of curated
            leaf samples. The backend scores each class, returns the top diagnosis, confidence score,
            and suggested actions based on agronomy best practices.
          </p>
        </div>
      </section>

      <section className="card bg-gradient-to-br from-primary-50 via-white to-emerald-50 border border-primary-100">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Impact highlights</h2>
        <div className="grid md:grid-cols-3 gap-6">
          {impactHighlights.map((item) => (
            <div key={item.title} className="p-4 rounded-xl bg-white shadow-sm border border-gray-100">
              <h3 className="font-semibold text-primary-700 mb-2">{item.title}</h3>
              <p className="text-sm text-gray-600">{item.description}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
};

export default About;

