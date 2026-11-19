import React, { useState } from 'react';
import confetti from 'canvas-confetti';
import { DETECTIVE_CASES } from '../services/detectiveCases';
import { HOTSPOT_REGIONS, EARTH_VARIABLES, generateCuratedTimeSeries } from '../services/nasaData';
import { analyzeTrend } from '../services/statistics';
import { DetectiveCase } from '../types/earthScience';
import { TimeSeriesChart } from './TimeSeriesChart';
import { SignificanceBadge } from './SignificanceBadge';
import { IrisGauge } from './IrisGauge';
import {
  Search,
  Award,
  HelpCircle,
  CheckCircle2,
  XCircle,
  ArrowRight,
  RotateCcw,
  ShieldCheck,
  FileText,
} from 'lucide-react';

function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return true;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

export const DetectiveMode: React.FC = () => {
  const [selectedCase, setSelectedCase] = useState<DetectiveCase>(DETECTIVE_CASES[0]);
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [quizAnswers, setQuizAnswers] = useState<{ [qIndex: number]: number }>({});
  const [showExplanations, setShowExplanations] = useState<{ [qIndex: number]: boolean }>({});
  const [caseCompleted, setCaseCompleted] = useState<boolean>(false);

  const region = HOTSPOT_REGIONS.find(r => r.id === selectedCase.regionId) || HOTSPOT_REGIONS[0];
  const variable = EARTH_VARIABLES.find(v => v.id === selectedCase.variableId) || EARTH_VARIABLES[0];

  const points = React.useMemo(() => {
    return generateCuratedTimeSeries(region, variable);
  }, [region, variable]);

  const stats = React.useMemo(() => {
    return analyzeTrend(points, variable.id, variable.name, variable.unit, region.name, region.coordinates, true);
  }, [points, variable, region]);

  const handleSelectAnswer = (qIndex: number, optionIndex: number) => {
    setQuizAnswers(prev => ({ ...prev, [qIndex]: optionIndex }));
    setShowExplanations(prev => ({ ...prev, [qIndex]: true }));
  };

  const isQuizPassed = selectedCase.guidingQuestions.every(
    (_, idx) => quizAnswers[idx] === selectedCase.guidingQuestions[idx].correctIndex
  );

  const handleCompleteCase = () => {
    setCaseCompleted(true);
    if (prefersReducedMotion()) return;
    try {
      confetti({
        particleCount: 120,
        spread: 80,
        origin: { y: 0.6 },
        colors: ['#D4AF37', '#F0D58A', '#F2F2F2', '#C0C0C0'],
      });
    } catch {
      // Fallback if canvas is blocked
    }
  };

  const handleResetCase = () => {
    setQuizAnswers({});
    setShowExplanations({});
    setCaseCompleted(false);
    setCurrentStep(1);
  };

  return (
    <div className="mx-auto max-w-7xl space-y-6">
      {/* Banner */}
      <div className="flex flex-col items-start justify-between gap-4 rounded-2xl border border-ink-700 bg-ink-900 p-6 md:flex-row md:items-center">
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1 rounded-full border border-gold-400/30 bg-ink-800 px-2.5 py-0.5 text-[11px] font-semibold text-gold-400">
              <Search className="h-3 w-3" />
              <span>Interactive student mission</span>
            </span>
            <span className="text-xs text-silver-500">Become a NASA data detective</span>
          </div>
          <h2 className="font-display text-2xl font-semibold tracking-tight text-silver-100">
            Earth science mystery dossiers
          </h2>
          <p className="max-w-2xl text-xs text-silver-300 sm:text-sm">
            Learn how space scientists separate seasonal noise from genuine climate trends, test statistical significance, and solve real-world planetary mysteries.
          </p>
        </div>

        <div className="flex w-full flex-wrap gap-2 md:w-auto">
          {DETECTIVE_CASES.map(c => (
            <button
              key={c.id}
              type="button"
              onClick={() => {
                setSelectedCase(c);
                handleResetCase();
              }}
              className={`rounded-xl px-3 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-none ${
                selectedCase.id === c.id
                  ? 'bg-gold-400 text-ink-950'
                  : 'border border-ink-700 bg-ink-950 text-silver-500 hover:bg-ink-800 hover:text-silver-100'
              }`}
            >
              {c.title}
            </button>
          ))}
        </div>
      </div>

      {/* Case header */}
      <div className="space-y-4 rounded-2xl border border-ink-700 bg-ink-900 p-6">
        <div className="flex flex-col justify-between gap-2 border-b border-ink-700 pb-3 sm:flex-row sm:items-center">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs font-semibold text-gold-400">
                Case #{DETECTIVE_CASES.indexOf(selectedCase) + 1}
              </span>
              <span
                className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                  caseCompleted
                    ? 'bg-silver-500/15 text-silver-500 border border-silver-500/30'
                    : 'bg-gold-400/15 text-gold-400 border border-gold-400/30'
                }`}
              >
                {caseCompleted ? 'Solved' : 'Open'}
              </span>
            </div>
            <span className="block text-xs font-semibold text-gold-400">
              {selectedCase.mysteryTitle}
            </span>
            <h3 className="mt-0.5 font-display text-xl font-semibold text-silver-100">{selectedCase.title}</h3>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded-lg border border-ink-700 bg-ink-950 px-2.5 py-1 text-silver-300">
              Target: <strong className="text-silver-100">{region.name}</strong>
            </span>
            <span className="rounded-lg border border-ink-700 bg-ink-950 px-2.5 py-1 text-gold-400">
              Variable: <strong>{variable.shortName}</strong>
            </span>
            <span className="rounded-lg border border-gold-400/30 bg-ink-800 px-2 py-1 font-semibold text-gold-300">
              {selectedCase.difficulty}
            </span>
          </div>
        </div>

        <p className="rounded-xl border border-ink-700 bg-ink-950/60 p-3.5 text-sm leading-relaxed text-silver-300 italic">
          &ldquo;{selectedCase.teaser}&rdquo;
        </p>

        <div className="grid grid-cols-3 gap-2 pt-2">
          <button
            type="button"
            onClick={() => setCurrentStep(1)}
            className={`rounded-xl border p-2.5 text-left transition-colors focus-visible:outline-none ${
              currentStep === 1
                ? 'border-gold-400/40 bg-gold-400/10 text-gold-400'
                : 'border-ink-700 bg-ink-950/60 text-silver-500'
            }`}
          >
            <span className="block text-[10px] text-silver-500">Step 1</span>
            <span className="block text-xs font-semibold">Briefing & suspects</span>
          </button>

          <button
            type="button"
            onClick={() => setCurrentStep(2)}
            className={`rounded-xl border p-2.5 text-left transition-colors focus-visible:outline-none ${
              currentStep === 2
                ? 'border-gold-400/40 bg-gold-400/10 text-gold-400'
                : 'border-ink-700 bg-ink-950/60 text-silver-500'
            }`}
          >
            <span className="block text-[10px] text-silver-500">Step 2</span>
            <span className="block text-xs font-semibold">Satellite evidence</span>
          </button>

          <button
            type="button"
            onClick={() => setCurrentStep(3)}
            className={`rounded-xl border p-2.5 text-left transition-colors focus-visible:outline-none ${
              currentStep === 3
                ? 'border-gold-400/40 bg-gold-400/10 text-gold-300'
                : 'border-ink-700 bg-ink-950/60 text-silver-500'
            }`}
          >
            <span className="block text-[10px] text-silver-500">Step 3</span>
            <span className="block text-xs font-semibold">Detective verdict</span>
          </button>
        </div>
      </div>

      {/* Step 1 */}
      {currentStep === 1 && (
        <div className="space-y-4 rounded-2xl border border-ink-700 bg-ink-900 p-6">
          <div className="space-y-3">
            <h4 className="flex items-center gap-2 font-display text-base font-semibold text-silver-100">
              <FileText className="h-5 w-5 text-gold-400" />
              <span>Case file</span>
            </h4>
          </div>

          <ol className="space-y-4 border-l-2 border-ink-700 pl-5">
            <li className="relative">
              <span className="absolute -left-[27px] top-1 flex h-5 w-5 items-center justify-center rounded-full bg-ink-950 border border-ink-700 text-[10px] font-semibold text-gold-400">
                1
              </span>
              <h5 className="text-xs font-semibold text-gold-400">Briefing</h5>
              <p className="mt-1 text-xs leading-relaxed text-silver-300 sm:text-sm">
                {selectedCase.briefing}
              </p>
            </li>
            <li className="relative">
              <span className="absolute -left-[27px] top-1 flex h-5 w-5 items-center justify-center rounded-full bg-ink-950 border border-ink-700 text-[10px] font-semibold text-gold-300">
                2
              </span>
              <h5 className="text-xs font-semibold text-gold-300">Orbital sensor clue</h5>
              <p className="mt-1 text-xs leading-relaxed text-silver-300">
                {selectedCase.satelliteClue}
              </p>
            </li>
            <li className="relative">
              <span className="absolute -left-[27px] top-1 flex h-5 w-5 items-center justify-center rounded-full bg-ink-950 border border-ink-700 text-[10px] font-semibold text-silver-300">
                3
              </span>
              <h5 className="text-xs font-semibold text-silver-300">Primary suspect drivers</h5>
              <ul className="mt-1 list-inside list-disc space-y-1.5 text-xs text-silver-300">
                {selectedCase.suspectDrivers.map((d, i) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            </li>
          </ol>

          <div className="flex justify-end pt-2">
            <button
              type="button"
              onClick={() => setCurrentStep(2)}
              className="flex items-center gap-2 rounded-xl bg-gold-400 px-5 py-2.5 text-xs font-semibold text-ink-950 transition-colors hover:bg-gold-400/90 focus-visible:outline-none"
            >
              <span>Examine satellite evidence</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* Step 2 */}
      {currentStep === 2 && (
        <div className="space-y-4">
          <div className="flex flex-col gap-4 lg:flex-row">
            <div className="flex-1">
              <TimeSeriesChart
                points={points}
                variable={variable}
                regionName={region.name}
                stats={stats}
              />
            </div>
            <div className="flex items-center justify-center rounded-2xl border border-ink-700 bg-ink-900 p-4">
              <IrisGauge
                pValue={stats.mkPValue}
                direction={
                  stats.direction === 'Increasing'
                    ? 'increasing'
                    : stats.direction === 'Decreasing'
                      ? 'decreasing'
                      : 'none'
                }
                size={96}
              />
            </div>
          </div>

          <div className="flex flex-col items-center justify-between gap-4 rounded-2xl border border-ink-700 bg-ink-900 p-4 sm:flex-row sm:p-5">
            <div className="text-xs text-silver-300">
              <span className="mb-0.5 block font-semibold text-silver-100">Detective guidance:</span>
              Look at the difference between the <strong className="text-silver-100">raw seasonal data</strong> and the <strong className="text-silver-100">deseasonalized anomaly</strong>. Note how the linear trend line slices right through!
            </div>
            <button
              type="button"
              onClick={() => setCurrentStep(3)}
              className="flex shrink-0 items-center gap-2 rounded-xl bg-gold-400 px-5 py-2.5 text-xs font-semibold text-ink-950 transition-colors hover:bg-gold-300 focus-visible:outline-none"
            >
              <span>Proceed to detective questions</span>
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      {/* Step 3 */}
      {currentStep === 3 && (
        <div className="space-y-6 rounded-2xl border border-ink-700 bg-ink-900 p-6">
          <div className="space-y-1">
            <h4 className="flex items-center gap-2 font-display text-base font-semibold text-silver-100">
              <HelpCircle className="h-5 w-5 text-gold-400" />
              <span>Detective verification questions</span>
            </h4>
            <p className="text-xs text-silver-500">
              Answer the questions below based on your observation of the satellite time-series.
            </p>
          </div>

          <div className="space-y-6">
            {selectedCase.guidingQuestions.map((q, qIndex) => {
              const selectedOption = quizAnswers[qIndex];
              const isAnswered = selectedOption !== undefined;
              const isCorrect = selectedOption === q.correctIndex;

              return (
                <div key={qIndex} className="space-y-3 rounded-xl border border-ink-700 bg-ink-950 p-4">
                  <div className="flex items-start justify-between">
                    <span className="text-xs font-semibold text-gold-400">
                      Question {qIndex + 1}
                    </span>
                    {isAnswered && (
                      <span className={`flex items-center gap-1 text-xs font-semibold ${isCorrect ? 'text-gold-400' : 'text-silver-500'}`}>
                        {isCorrect ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
                        <span>{isCorrect ? 'Correct evidence' : 'Review clue'}</span>
                      </span>
                    )}
                  </div>

                  <p className="text-sm font-semibold text-silver-100">{q.question}</p>

                  <div className="grid grid-cols-1 gap-2 pt-1 sm:grid-cols-2">
                    {q.options.map((opt, optIdx) => {
                      let btnStyle = 'bg-ink-900 hover:bg-ink-800 text-silver-300 border-ink-700';
                      if (isAnswered) {
                        if (optIdx === q.correctIndex) {
                          btnStyle = 'bg-gold-400/10 border-gold-400 text-gold-300 font-semibold';
                        } else if (optIdx === selectedOption) {
                          btnStyle = 'bg-silver-500/10 border-silver-500 text-silver-500';
                        }
                      }

                      return (
                        <button
                          key={optIdx}
                          type="button"
                          onClick={() => handleSelectAnswer(qIndex, optIdx)}
                          disabled={caseCompleted}
                          className={`rounded-xl border p-3 text-left text-xs transition-colors focus-visible:outline-none disabled:opacity-60 ${btnStyle}`}
                        >
                          {opt}
                        </button>
                      );
                    })}
                  </div>

                  {showExplanations[qIndex] && (
                    <div className="rounded-lg border border-gold-400/30 bg-gold-400/5 p-3 text-xs leading-relaxed text-silver-300">
                      <strong className="text-silver-100">Detective note:</strong> {q.explanation}
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {isQuizPassed && !caseCompleted && (
            <div className="flex flex-col items-center justify-between gap-4 rounded-2xl border border-silver-500/40 bg-ink-950 p-5 sm:flex-row">
              <div>
                <h5 className="text-sm font-semibold text-silver-100">All clues verified. Ready to issue the scientific verdict?</h5>
                <p className="text-xs text-silver-500">Your mathematical analysis matches peer-reviewed satellite observations.</p>
              </div>
              <button
                type="button"
                onClick={handleCompleteCase}
                className="flex items-center gap-2 rounded-xl bg-silver-500 px-6 py-3 text-xs font-semibold text-ink-950 transition-colors hover:bg-silver-300 focus-visible:outline-none"
              >
                <Award className="h-4 w-4" />
                <span>Issue official verdict</span>
              </button>
            </div>
          )}

          {caseCompleted && (
            <div className="space-y-4 rounded-2xl border border-silver-500/50 bg-ink-950 p-6">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-silver-500 p-3 text-ink-950">
                  <Award className="h-8 w-8" />
                </div>
                <div>
                  <span className="block text-[10px] font-semibold text-silver-500">
                    NASA Earth science investigation resolved
                  </span>
                  <h4 className="font-display text-lg font-semibold text-silver-100">
                    Official detective certification: {selectedCase.title}
                  </h4>
                </div>
              </div>

              <div className="space-y-2 rounded-xl border border-ink-700 bg-ink-900 p-4 text-xs leading-relaxed text-silver-300">
                  <div className="flex items-center justify-between border-b border-ink-700 pb-2">
                    <span className="text-silver-500">Statistical significance:</span>
                    <IrisGauge
                      pValue={stats.mkPValue}
                      direction={
                        stats.direction === 'Increasing'
                          ? 'increasing'
                          : stats.direction === 'Decreasing'
                            ? 'decreasing'
                            : 'none'
                      }
                      size={64}
                    />
                  </div>

                <div>
                  <strong className="mb-1 block text-silver-100">Debrief verdict:</strong>
                  {selectedCase.verdictDebrief}
                </div>
                <div className="border-t border-ink-700 pt-2">
                  <strong className="mb-1 block text-silver-100">Planetary consequence:</strong>
                  {selectedCase.realWorldImpact}
                </div>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-2 pt-2">
                <button
                  type="button"
                  onClick={handleResetCase}
                  className="flex items-center gap-1.5 rounded-lg bg-ink-800 px-3 py-1.5 text-xs font-medium text-silver-300 transition-colors hover:bg-ink-700 focus-visible:outline-none"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  <span>Re-examine case</span>
                </button>

                <div className="flex items-center gap-1 text-xs text-silver-500">
                  <ShieldCheck className="h-4 w-4" />
                  <span>Dossier verified with NASA Earth science data standards</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
